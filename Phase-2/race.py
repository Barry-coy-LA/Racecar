import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
from numba import njit
from pyglet.gl import GL_POINTS
from PP_control import PurePursuit_Controller
from MPC_control import MPC_Controller
from Stanley_control import Stanley_Controller
import csv


# ctr = "PurePursuit"
ctr = "Stanley"

# MPC does not work. this model has huge error
# ctr = "MPC"

class envoriment:
    def __init__(self, conf):
        self.conf = conf
        self.load_waypoints(conf)
        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

def main():
    with open('config_Silverstone_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    envor = envoriment(conf)
    if (ctr == "PurePursuit"):
        planner = PurePursuit_Controller(conf, (0.17145+0.15875))
        work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 1.442805378751944, 'vgain': 0.73}#0.90338203837889}
    elif (ctr == "MPC"):
        planner = MPC_Controller(conf, (0.17145+0.15875))
    elif (ctr == "Stanley"):
        planner = Stanley_Controller(conf, (0.17145+0.15875))
        work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 1.482805378751944, 'vgain': 0.891}#0.90338203837889}
    def render_callback(env_renderer):
        # custom extra drawing function
        e = env_renderer
        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        envor.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env.add_render_callback(render_callback)
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    laptime = 0.0
    start = time.time()
    steer = 0
    speed = 0
    csv_file = f'output_{ctr}.csv'
    with open(csv_file, mode='w', newline='') as file:
       writer = csv.writer(file)
       writer.writerow(['Laptime', 'Speed', 'Steering Angle', 'head', 'Lateral Error'])

    while not done:
        if (ctr == "PurePursuit"):
            speed, steer, error_front_axle = planner.controller(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        elif (ctr == "MPC"):
            error_front_axle = []
            speed, steer = planner.controller(obs['poses_x'][0], obs['poses_y'][0] , speed, obs['poses_theta'][0])
        elif (ctr == "Stanley"):
            speed, steer, error_front_axle = planner.controller(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
            # Save the data to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([laptime, speed, steer, error_front_axle])
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()
