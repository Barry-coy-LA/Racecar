import numpy as np
import random
import copy
import os
import argparse
import yaml
import time
from world import World
from robot import Robot
from motion_model import MotionModel
from measurement_model import MeasurementModel
from utils import *
from icp import icp_matching
from sklearn.neighbors import KDTree
import concurrent.futures
NUMBER_OF_PARTICLES = 5
PREPROCESS_CONV_SIZE = 7
radar_length = 220

def custom_binary_search(arr, target):

    left, right = 0, len(arr) - 1
    result_index = -1 
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            result_index = mid
            left = mid + 1
        else:
            right = mid - 1

    return result_index

def voxel_grid_downsample(points, voxel_size):
    """
    对点云进行体素网格下采样
    :param points: 输入的点云数组 (N, 2) 或 (N, 3)
    :param voxel_size: 体素大小，用于控制下采样的精度
    :return: 下采样后的点云数组
    """
    # 计算每个点所在的体素索引
    if len(points) == 0:
        return np.array([])
    min_coords = np.min(points, axis=0)
    voxel_indices = ((points - min_coords) // voxel_size).astype(np.int32)

    # 使用字典来存储每个体素中的点（仅保留一个代表点）
    voxel_dict = {}
    for i, voxel_index in enumerate(voxel_indices):
        voxel_key = tuple(voxel_index)
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = points[i]

    # 获取下采样后的点
    downsampled_points = np.array(list(voxel_dict.values()))
    return downsampled_points

class SLAM:
    def __init__(self, config, conf, conf2):
        self.config = config
        self.world = World()
        self.map = conf.map_path + conf.map_ext
        self.world.read_map(self.map)
        self.output_path = None

        self.R_init = [conf.sx, conf.sy, conf.stheta]
        self.p_init = [conf.sx, conf.sy, conf.stheta]
        self.NUMBER_OF_MODE_SAMPLES = self.config['num_mode_samples']
        self.MODE_SAMPLE_COV = np.diag(self.config['mode_sample_cov'])
        self.w = [1 / NUMBER_OF_PARTICLES] * NUMBER_OF_PARTICLES
        self.p = [None] * NUMBER_OF_PARTICLES

        self.motion_model = None 
        self.measurement_model =  None

        self.ROBOT = self.config['robot']
        self.prev_odo = []

        self.num_sensors = self.ROBOT['num_sensors']
        self.radians_per_elem = None
        self.MAX_LIDAR_DIST = self.ROBOT['radar_range']
        self.resolution = conf2.resolution
        self.orign = conf2.origin
        self.world_grid = self.world.get_grid()
        self.world_size = self.world.get_size()
        self.count = 0
        self.slamtime_total = 0
 
    def fastslam_init(self):
        self.output_path = self.config['output_path']
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.output_path = os.path.join(self.output_path, "fastslam2")
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.output_path = os.path.join(self.output_path, 'map')

        (x, y, theta) = self.real2map_scale(self.R_init)    

        self.R = Robot(x, y, degree2radian(theta), self.world_grid, self.ROBOT, sense_noise=3.0)

        self.prev_odo = self.R.get_state()  ##

        (x, y, theta) = self.real2map_scale(self.p_init)
        (height, width)=self.world.get_size()
        init_grid = np.ones([height,width]) * self.ROBOT['prior_prob']

        for i in range(NUMBER_OF_PARTICLES):
            self.p[i] = Robot(x, y, degree2radian(theta), copy.deepcopy(init_grid), self.ROBOT)

        self.motion_model = MotionModel(self.config['motion_model'])  

        self.measurement_model = MeasurementModel(self.config['measurement_model'], self.ROBOT['radar_range'])

    def real2map_scale(self,state):
        state[0] = (state[0]-self.orign[0])/self.resolution
        state[1] = (state[1]-self.orign[1])/self.resolution
        return state


    def getFree_Occupy(self,z_star, state):
        indices = np.linspace(0, len(z_star) - 1, self.num_sensors, dtype=int)
        z_star = np.convolve(z_star, np.ones(PREPROCESS_CONV_SIZE), 'same') / PREPROCESS_CONV_SIZE
        z_star = z_star[indices]
        x = (state[0] - self.orign[0])/self.resolution
        y = (state[1] - self.orign[1])/self.resolution
        radar_src = np.array([[x] * self.num_sensors, [y] * self.num_sensors])
        
        theta = state[2]
        free_grid, occupy_grid = [], []

        z_star = np.clip(z_star, 0, self.MAX_LIDAR_DIST)
        self.radar_theta = np.arange(0, self.num_sensors) * (4.7 / len(z_star)) - 2.35
        radar_theta = self.radar_theta + theta
        radar_rel_dest = np.stack(
            (
                np.cos(radar_theta) * radar_length,
                np.sin(radar_theta) * radar_length
            ), axis=0
        )
        radar_dest = radar_rel_dest + radar_src

        beams = [None] * self.num_sensors
        for i in range(self.num_sensors):
            x1, y1 = radar_src[:, i]
            x2, y2 = radar_dest[:, i]
            beams[i] = bresenham(x1, y1, x2, y2, self.world_size[0], self.world_size[1])
            beam = beams[i]
            z_star[i] /= self.resolution

            if z_star[i] > radar_length + 1:
                free_grid.extend(list(beam))
            else:
                dist = np.linalg.norm(beam - radar_src[:,1], axis=1)
                idx = custom_binary_search(dist, z_star[i])
                occupy_grid.append(list(beam[idx]))
                free_grid.extend(list(beam[:idx]))

        return z_star, free_grid, occupy_grid
    
    def process_particle(self, i, curr_odo, z_star, free_grid_offset_star, occupy_grid_offset_star):
        prev_pose = self.p[i].get_state()
        tmp_r = Robot(0, 0, 0, self.p[i].grid, self.ROBOT)
        guess_pose = self.motion_model.sample_motion_model(self.prev_odo, curr_odo, prev_pose)
        scan = relative2absolute(occupy_grid_offset_star, guess_pose).astype(np.int32)

        tmp = np.where(self.p[i].grid >= 0.9)
        edges = np.stack((tmp[1], tmp[0])).T
        voxel_size = 2
        downsampled_edges = voxel_grid_downsample(edges, voxel_size)

        pose_hat = icp_matching(downsampled_edges, scan, guess_pose)
        if pose_hat is not None:
            samples = np.random.multivariate_normal(pose_hat, self.MODE_SAMPLE_COV, self.NUMBER_OF_MODE_SAMPLES) 
            likelihoods = np.zeros(self.NUMBER_OF_MODE_SAMPLES)
 
            for j in range(self.NUMBER_OF_MODE_SAMPLES):
                motion_prob = self.motion_model.motion_model(self.prev_odo, curr_odo, prev_pose, samples[j])
                x, y, theta = samples[j]
                tmp_r.set_states(x, y, theta)
                z, _, _ = tmp_r.sense()
                measurement_prob = self.measurement_model.measurement_model(z_star, z)
                likelihoods[j] = motion_prob * measurement_prob
                
            eta = np.sum(likelihoods)
            if eta > 0:
                pose_mean = np.sum(samples * likelihoods[:, np.newaxis], axis=0) / eta
                tmp = samples - pose_mean
                pose_cov = tmp.T @ (tmp * likelihoods[:, np.newaxis]) / eta
                new_pose = np.random.multivariate_normal(pose_mean, pose_cov, 1)
                x, y, theta = new_pose[0]
                self.p[i].set_states(x, y, theta)
            self.w[i] *= eta
            print('scan')
        else:
            x, y, theta = self.motion_model.sample_motion_model(self.prev_odo, curr_odo, prev_pose)
            self.p[i].set_states(x, y, theta)
            z, _, _ = self.p[i].sense()
            self.w[i] *= self.measurement_model.measurement_model(z_star, z)
            print('not scan')
        self.p[i].update_trajectory()
        curr_pose = self.p[i].get_state()
        free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
        occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
        self.p[i].update_occupancy_grid(free_grid, occupy_grid)
        return self.p[i], self.w[i]
                

    def fastslam(self, state, z_star, step):
        start1 = time.time()
        curr_odo = np.array(state)
        curr_odo = self.real2map_scale(curr_odo)
        self.R.set_states(curr_odo[0],curr_odo[1],curr_odo[2])
        self.R.update_trajectory()
        z_star, free_grid_star, occupy_grid_star = self.getFree_Occupy(z_star, state)
        # print(np.array(occupy_grid_star).shape)
        # print('occupy_grid_star:',occupy_grid_star)

        free_grid_offset_star = absolute2relative(free_grid_star, curr_odo)
        occupy_grid_offset_star = absolute2relative(occupy_grid_star, curr_odo)

        for i in range(NUMBER_OF_PARTICLES):
            # print('1-------------------------------1')
            prev_pose = self.p[i].get_state()
            # print('self.prev_odo:',self.prev_odo)
            # print('curr_odo:',curr_odo)
            # print('prev_pose:',prev_pose)

            tmp_r = Robot(0, 0, 0, self.p[i].grid, self.ROBOT)
            # print('2-------------------------------2')
            # generate initial guess from motion model
            guess_pose = self.motion_model.sample_motion_model(self.prev_odo, curr_odo, prev_pose, step)
            # print('guess_pose:',guess_pose)
            # print('3-------------------------------3')
            scan = relative2absolute(occupy_grid_offset_star, guess_pose).astype(np.int32) #世界坐标系
            # print('scan:',scan)

            tmp = np.where(self.p[i].grid >= 0.9)
            edges = np.stack((tmp[1], tmp[0])).T

            # print('edges:',np.array(edges).shape)
            voxel_size = 3  # 设置体素网格大小，可以根据需要调整
            downsampled_edges = voxel_grid_downsample(edges, voxel_size)
            # print('downsampled_edges:',np.array(downsampled_edges).shape)
            # refine guess pose by scan matching
            start = time.time()
            pose_hat = icp_matching(downsampled_edges, scan, guess_pose)
            # print('icp spend time:',time.time()-start)

            # start = time.time()
            # If the scan matching fails, the pose and the weights are computed according to the motion model
            if pose_hat is not None:
                # Sample around the mode
                samples= np.random.multivariate_normal(pose_hat, self.MODE_SAMPLE_COV, self.NUMBER_OF_MODE_SAMPLES) 
                # Compute gaussain proposal
                likelihoods = np.zeros(self.NUMBER_OF_MODE_SAMPLES)
 
                for j in range(self.NUMBER_OF_MODE_SAMPLES):
                    motion_prob = self.motion_model.motion_model(self.prev_odo, curr_odo, prev_pose, samples[j], step)

                    x, y, theta = samples[j]
                    tmp_r.set_states(x, y, theta)
                    z, _, _ = tmp_r.sense()
                    measurement_prob = self.measurement_model.measurement_model(z_star, z)

                    likelihoods[j] = motion_prob * measurement_prob
                    
                eta = np.sum(likelihoods)
                if eta > 0:
                    pose_mean = np.sum(samples * likelihoods[:, np.newaxis], axis=0)
                    pose_mean = pose_mean / eta

                    tmp = samples - pose_mean
                    pose_cov = tmp.T @ (tmp * likelihoods[:, np.newaxis])
                    pose_cov = pose_cov / eta

                    # Sample new pose of the particle from the gaussian proposal
                    new_pose = np.random.multivariate_normal(pose_mean, pose_cov, 1)
                    x, y, theta = new_pose[0]
                    self.p[i].set_states(x, y, theta)

                # Update weight
                self.w[i] *= eta
                print("scan: ")
                
            else:
                # Simulate a robot motion for each of these particles
                x, y, theta = self.motion_model.sample_motion_model(self.prev_odo, curr_odo, prev_pose)
                self.p[i].set_states(x, y, theta)
        
                # Calculate particle's weights depending on robot's measurement
                z, _, _ = self.p[i].sense()
                self.w[i] *= self.measurement_model.measurement_model(z_star, z)
                print("not scan: ")

            self.p[i].update_trajectory()
            # print('praticle spend time:',time.time()-start)
            # print('4-------------------------------4')

            # Update occupancy grid based on the true measurements
            curr_pose = self.p[i].get_state()
            free_grid = relative2absolute(free_grid_offset_star, curr_pose).astype(np.int32)
            occupy_grid = relative2absolute(occupy_grid_offset_star, curr_pose).astype(np.int32)
            self.p[i].update_occupancy_grid(free_grid, occupy_grid)  

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self.process_particle, i, curr_odo, z_star, free_grid_offset_star, occupy_grid_offset_star) for i in range(NUMBER_OF_PARTICLES)]
        #     results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # for i, (particle, weight) in enumerate(results):
        #     self.p[i] = particle
        #     self.w[i] = weight
        
        # normalize·
        # print('5-------------------------------5')
        # print('self.w:',self.w)
        self.w = self.w / np.sum(self.w)
        best_id = np.argsort(self.w)[-1]
        # print('normalize self.w:',self.w)
        # select best particle
        estimated_R = copy.deepcopy(self.p[best_id])

        # adaptive resampling
        N_eff = 1 / np.sum(self.w ** 2)
        # print('N_eff:',N_eff)

        # print('6-------------------------------6')
        if N_eff < NUMBER_OF_PARTICLES / 2:
            # print("Resample!")
            # Resample the particles with a sample probability proportional to the importance weight
            # Use low variance sampling method
            new_p = [None] * NUMBER_OF_PARTICLES
            new_w = [None] * NUMBER_OF_PARTICLES
            J_inv = 1 / NUMBER_OF_PARTICLES
            r = random.random() * J_inv
            c = self.w[0]

            i = 0
            for j in range(NUMBER_OF_PARTICLES):
                U = r + j * J_inv
                while (U > c):
                    i += 1
                    c += self.w[i]
                new_p[j] = copy.deepcopy(self.p[i])
                new_w[j] = self.w[i]

            self.p = new_p
            self.w = new_w

        self.prev_odo = curr_odo
        self.count += 1
        slamtime = time.time()-start1
        
        print('Slam time:',slamtime)
        self.slamtime_total += slamtime
        avg = self.slamtime_total/self.count
        print('avg_time:',avg)
        

        visualize(self.R, self.p, estimated_R, free_grid_star, self.count, "FastSLAM 2.0", self.output_path, False)
