import random
import numpy as np
import time
from utils import *
from concurrent.futures import ThreadPoolExecutor

def process_beam(beam, loc, grid):
    beam = np.array(beam)
    dist = np.linalg.norm(beam - loc, axis=1)
    grid_values = grid[beam[:, 1], beam[:, 0]]
    obstacle_positions = np.where(grid_values >= 0.9)[0]
    if len(obstacle_positions) > 0:
        idx = obstacle_positions[0]
        return dist[idx], beam[:idx], beam[idx]
    else:
        return None, beam, None

class Robot(object):
    def __init__(self, x, y, theta, grid, config, sense_noise=None):
        # initialize robot pose
        self.x = x
        self.y = y
        self.theta = theta
        self.trajectory = []

        # map that robot navigates in
        # for particles, it is a map with prior probability
        self.grid = grid
        self.grid_size = self.grid.shape

        # probability for updating occupancy map
        self.prior_prob = config['prior_prob']
        self.occupy_prob = config['occupy_prob']
        self.free_prob = config['free_prob']

        # sensing noise for trun robot measurement
        self.sense_noise = sense_noise if sense_noise is not None else 0.0

        # parameters for beam range sensor
        self.num_sensors = config['num_sensors']
        self.radar_theta = np.arange(0, self.num_sensors) * (4.7 / self.num_sensors) - 2.35
        self.radar_length = config['radar_length']
        self.radar_range = config['radar_range']

    def set_states(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def get_state(self):
        return (self.x, self.y, self.theta)
    
    def update_trajectory(self):
        self.trajectory.append([self.x, self.y])

    def sense(self, world_grid=None):
        # start = time.time()
        measurements, free_grid, occupy_grid = self.ray_casting(world_grid)
        measurements = np.clip(measurements + np.random.normal(0.0, self.sense_noise, self.num_sensors), 0.0, self.radar_range)
        # print('sense time:',time.time()-start)
        return measurements, free_grid, occupy_grid

    def build_radar_beams(self):
        radar_src = np.array([[self.x] * self.num_sensors, [self.y] * self.num_sensors])
        
        radar_theta = self.radar_theta + self.theta
        
        radar_rel_dest = np.stack(
            (
                np.cos(radar_theta) * self.radar_length,
                np.sin(radar_theta) * self.radar_length
            ), axis=0
        )
        radar_dest = radar_rel_dest + radar_src

        beams = [None] * self.num_sensors
        for i in range(self.num_sensors):
            x1, y1 = radar_src[:, i]
            x2, y2 = radar_dest[:, i]
            beams[i] = bresenham(x1, y1, x2, y2, self.grid_size[0], self.grid_size[1])

        return beams
    
    def ray_casting(self, world_grid=None):
        # start = time.time()
        beams = self.build_radar_beams()
        # print('beams_time:',time.time()-start)

        loc = np.array([self.x, self.y])    #Car position
        measurements = [self.radar_range] * self.num_sensors
        free_grid, occupy_grid = [], []

        # start1 = time.time()
        # 使用并行处理
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda beam: process_beam(beam, loc, self.grid), beams))

        # 处理并行返回的结果
        for i, result in enumerate(results):
            dist, free_part, obstacle_point = result
            free_grid.extend(list(free_part))
            if obstacle_point is not None:
                occupy_grid.append(list(obstacle_point))
                measurements[i] = dist
        # print('loop:',time.time()-start1)

        return measurements, free_grid, occupy_grid
    
    def update_occupancy_grid(self, free_grid, occupy_grid):
        mask1 = np.logical_and(0 < free_grid[:, 0], free_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < free_grid[:, 1], free_grid[:, 1] < self.grid_size[0])
        free_grid = free_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(False)
        l = prob2logodds(self.grid[free_grid[:, 1], free_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[free_grid[:, 1], free_grid[:, 0]] = logodds2prob(l)

        mask1 = np.logical_and(0 < occupy_grid[:, 0], occupy_grid[:, 0] < self.grid_size[1])
        mask2 = np.logical_and(0 < occupy_grid[:, 1], occupy_grid[:, 1] < self.grid_size[0])
        occupy_grid = occupy_grid[np.logical_and(mask1, mask2)]

        inverse_prob = self.inverse_sensing_model(True)
        l = prob2logodds(self.grid[occupy_grid[:, 1], occupy_grid[:, 0]]) + prob2logodds(inverse_prob) - prob2logodds(self.prior_prob)
        self.grid[occupy_grid[:, 1], occupy_grid[:, 0]] = logodds2prob(l)
    
    def inverse_sensing_model(self, occupy):
        if occupy:
            return self.occupy_prob
        else:
            return self.free_prob
