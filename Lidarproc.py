import numpy as np

class WallFollower:
    PREPROCESS_CONV_SIZE = 3
    MAX_LIDAR_DIST = 300    
    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None

    def preprocess_lidar(self, ranges):
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # Removw the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # Sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges
    
    def compute_right_distance(self,pro_ranges,perpend_line_ind,one_line_ind):
        # Compute the distance between wall and right side of car
        ar = pro_ranges[one_line_ind]       # Small angle alph
        br = pro_ranges[perpend_line_ind]   # Perpendicular to the central axis of the car
        alph = (one_line_ind-perpend_line_ind) * self.radians_per_elem
        theta = np.arctan(np.add(ar*np.cos(alph),-br)/(ar*np.sin(alph)))
        print(f"r_theta: {theta}")
        right_distance = br*np.cos(theta)
        return theta,right_distance
    
    def compute_left_distance(self,pro_ranges,perpend_line_ind,one_line_ind):
        # compute the distance between wall and left side of car
        al = pro_ranges[one_line_ind]       # Small angle alph
        bl = pro_ranges[perpend_line_ind]   # Perpendicular to the central axis of the car
        alph = (one_line_ind-perpend_line_ind) * self.radians_per_elem
        print(self.radians_per_elem)
        theta = np.arctan(np.add(al*np.cos(alph),-bl)/(al*np.sin(alph)))
        left_distance = bl*np.sin(theta)
        print(f"l_theta: {theta}")
        theta = np.pi/4 - theta
        return theta,left_distance

    def process_lidar(self, ranges):
        # Preprocess the Lidar Information
        proc_ranges = self.preprocess_lidar(ranges)
        # Find car centerline to LiDAR
        center_index = int(np.floor(len(proc_ranges)/2))
        # Find the right distance
        r_perpend_line_index = int(np.floor(center_index + np.pi/2/self.radians_per_elem))
        r_one_line_index = int(np.floor(r_perpend_line_index - 20 * np.pi/180/self.radians_per_elem))
        theta,right_distance = self.compute_right_distance(proc_ranges,r_perpend_line_index,r_one_line_index)
        # Find the left distance
        l_perpend_line_index = int(np.floor(center_index - np.pi/2/self.radians_per_elem))
        l_one_line_index = int(np.floor( l_perpend_line_index - 20 * np.pi/180/self.radians_per_elem))
        theta2, left_distance = self.compute_left_distance(proc_ranges,l_perpend_line_index,l_one_line_index)
        # Find the desired path
        
        desired_distance = (left_distance+right_distance)/2
        # Find the desired steer
        desired_theta = (theta+theta2)/2
        # Display the result
        print(f"left_distance: {left_distance}")
        print(f"rigth_distance: {right_distance}")
        print(f"desired_distance: {desired_distance}")
        print(f"theta: {theta}")
        # Send back the distance and steering angle to the simulation
        return desired_theta, right_distance, desired_distance

