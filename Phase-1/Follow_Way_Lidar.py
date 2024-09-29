import time
import gym
import numpy as np
import concurrent.futures
import os
import sys
import yaml
from argparse import Namespace
import csv

# Get ./src/ folder & add it to path
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# Import drivers
from Lidarproc import WallFollower
drivers = [WallFollower()]
# Choose racetrack
RACETRACK = 'Monza'

def control(theta, right_distance, desired_distance, step_reward, previous_error, error_integral):
    def steering_constraint_tanh(x):
        # Use the function tanh() to limit the max steering
        return 18*np.pi*np.tanh(x)/180
    # Parameter
    vgain = 1.375
    K = 0.70  # Compensation coefficient of car width
    speed = 6
    lamda = 0.0018 # Coefficient of variation in speed
    speed_c = vgain * speed
    L = step_reward * speed
    y = desired_distance - K * right_distance
    error_term = - y - L*np.sin(theta)
    error_integral += error_term
    Kd = 0.034
    Kp = 1.87
    Ki = 0.03
    # PID
    steering_angle1 = Kp * error_term + Kd * (error_term - previous_error) + Ki * error_integral
    steering_angle = steering_constraint_tanh(steering_angle1)
    speed_c = speed_c * np.power(lamda, abs(steering_angle))
    return speed_c, steering_angle, steering_angle1

def render_callback(env_renderer):
    # custom extra drawing function
    e = env_renderer
    camera_coverage_side_length = 800
    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - camera_coverage_side_length
    e.right = right + camera_coverage_side_length
    e.top = top + camera_coverage_side_length
    e.bottom = bottom - camera_coverage_side_length

class GymRunner(object):
    def __init__(self, racetrack, drivers):
        self.racetrack = racetrack
        self.drivers = drivers
        self.speed_data = []
        self.laptime_data = []
        # self.csv_file = 'simulation_data.csv'
        # Initialize the CSV file and write the header
        # with open(self.csv_file, mode='w', newline='') as file:
        #    writer = csv.writer(file)
        #    writer.writerow(['Laptime', 'Speed', 'Steering Angle','Steering', 'Theta', 'Right Distance', 'Desired Distance'])

    def run(self):
        # Load map and specific information for the map
        with open('config_example_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)

        # Create the F1TENTH GYM environment
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        env.add_render_callback(render_callback)  # Camera by following

        # Initially parametrize the environment
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        env.render()
        laptime = 0.0
        start = time.time()
        steer = 0
        error_int = 0
        count = 0
        speed_re = 0
        # Start the simulation
        while not done:
            actions = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, driver in enumerate(drivers):
                    # Proc Lidar data
                    futures.append(executor.submit(driver.process_lidar, obs['scans'][i]))
            for future in futures:
                # result of proc_Lidar_data
                theta, right_distance, desired_distance = future.result() 
                # Controller
                speed, steer, steering= control(theta, right_distance, desired_distance, step_reward, steer, error_int)
                actions.append([steer, speed])
                speed_re += speed
                # Save the data to CSV
                # with open(self.csv_file, mode='a', newline='') as file:
                #    writer = csv.writer(file)
                #    writer.writerow([laptime, speed, steer, steering, theta, right_distance, desired_distance])

            actions = np.array(actions)
            # Send the actions from the Follow the Gap to the simulation environment
            obs, step_reward, done, info = env.step(actions)
            laptime += step_reward
            count += 1
            speed_mean = speed_re/count
            env.render(mode='human')
            print(f"avarage speed: {speed_mean}")
            print(f"--------------------------")
        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

if __name__ == '__main__':
    runner = GymRunner(RACETRACK, drivers)
    runner.run()
