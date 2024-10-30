import numpy as np
from utils import wrapAngle, normalDistribution
import math


class MotionModel(object):
    def __init__(self, config):
        self.alpha1 = config['alpha1']
        self.alpha2 = config['alpha2']
        self.r_len = 0.17145
        self.f_len = 0.15875

    def sample_motion_model(self, prev_odo, curr_odo, prev_pose, dt=0.01):
        x, y, psi = prev_pose

        # Computer the speed and steering
        delta_x = curr_odo[0] - prev_odo[0]
        delta_y = curr_odo[1] - prev_odo[1]

        velocity = np.sqrt(delta_x**2 + delta_y**2) / dt
        beta = math.atan2(delta_y, delta_x) - psi
        beta = wrapAngle(beta)
        steering = math.atan((self.f_len + self.r_len) / self.r_len * math.tan(beta))
        # Add noise
        noisy_velocity = velocity + np.random.normal(0, self.alpha1 + self.alpha2)
        noisy_steering = steering + np.random.normal(0, self.alpha1 + self.alpha2)

        # Compute the beta
        beta = math.atan((self.r_len / (self.r_len + self.f_len)) * math.tan(noisy_steering))
        # Update
        new_x = x + noisy_velocity * math.cos(psi + beta) * dt

        new_y = y + noisy_velocity * math.sin(psi + beta) * dt

        new_psi = psi + (noisy_velocity / self.f_len) * math.sin(beta) * dt

        return new_x, new_y, new_psi


    def motion_model(self, prev_odo, curr_odo, prev_pose, curr_pose, dt=0.01):
        delta_x = curr_odo[0] - prev_odo[0]
        delta_y = curr_odo[1] - prev_odo[1]
        # delta_psi = wrapAngle(curr_odo[2] - prev_odo[2])

        velocity = np.sqrt(delta_x**2 + delta_y**2) / dt
        beta = math.atan2(delta_y, delta_x) -prev_odo[2]
        beta = wrapAngle(beta)
        steering = math.atan((self.f_len + self.r_len) / self.r_len * math.tan(beta))

        delta_x_prime = curr_pose[0] - prev_pose[0]
        delta_y_prime = curr_pose[1] - prev_pose[1]
        # delta_psi_prime = wrapAngle(curr_pose[2] - prev_pose[2])

        velocity_prime = np.sqrt(delta_x_prime**2 + delta_y_prime**2) / dt
        beta_prime = math.atan2(delta_y_prime, delta_x_prime) - prev_pose[2]
        beta_prime = wrapAngle(beta_prime)
        steering_prime = math.atan((self.f_len + self.r_len) / self.r_len * math.tan(beta_prime))

        p1 = normalDistribution(velocity-velocity_prime, self.alpha1 * velocity_prime ** 2 + self.alpha2)
        p2 = normalDistribution(wrapAngle(steering - steering_prime), self.alpha1 * steering_prime ** 2 + self.alpha2)
        
        return p1*p2

