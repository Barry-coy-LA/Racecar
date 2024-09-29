import numpy as np
from PP_control import nearest_point_on_trajectory
from PP_control import first_point_on_trajectory_intersecting_circle

class Stanley_Controller:
    def __init__(self, conf, wb):
        self.k: int = 7.5
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((4, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            current_waypoint[3] = waypoints[i, self.conf.wpt_thind]
            return current_waypoint, nearest_point, nearest_dist
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind], waypoints[i, self.conf.wpt_thind]), nearest_point, nearest_dist
        else:
            return None
        
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def cross_product_2d(self, a, b):
        return a[0] * b[1] - a[1] * b[0]
        
    def controller(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point, nearest_waypoint, error_front_axle = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)
        vec1 = position - nearest_waypoint
        vec2 = nearest_waypoint - lookahead_point[0:2]
        cross_prod = self.cross_product_2d(vec2, vec1)
        if cross_prod > 0:
            error_front_axle = error_front_axle
        elif cross_prod < 0:
            error_front_axle = -error_front_axle
        # theta_e corrects the heading error
        theta_e = self.normalize_angle(lookahead_point[3] - pose_theta)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, lookahead_point[2])
        # Steering control
        steering_angle = theta_e + theta_d
        # print("control is running")
        speed = lookahead_point[2]
        speed = vgain * speed

        return speed, steering_angle, error_front_axle
