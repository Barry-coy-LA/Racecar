import pandas as pd
import numpy as np
from scipy.ndimage import median_filter

# Vehicle parameters
car_mass = 3.463388126201571  # Mass of the car
mu = 1.0489  # Friction coefficient between tire and ground (assumed value)
g = 9.81  # Gravitational acceleration

# Calculate maximum lateral and longitudinal accelerations
a_y_max = mu * g  # Maximum lateral acceleration
a_x_max = 12.0  # Maximum allowed longitudinal acceleration (m/s²)
max_velocity = 9
min_velocity = 3
MAP_NAME = "Silverstone_map"

# Read CSV file
file_path = f"maps/{MAP_NAME}_secondplanning.csv" 
waypoints_df = pd.read_csv(file_path)
waypoints_df.columns = waypoints_df.columns.str.strip('# ')

# Extract x and y coordinates
x = waypoints_df['x_m'].values
y = waypoints_df['y_m'].values

# Initialize lists
s_m = [0]  # Cumulative distance
curvature = []
psi_rad = []
vx_mps = []
ax_mps2 = [0] * (len(x))  # Initialize with zeros for correct length
ax_mps3 = [0] * (len(x))  

# Function to calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Calculate cumulative distance, curvature, and heading angle
for i in range(1, len(x)):
    s_m.append(s_m[-1] + euclidean_distance(x[i-1], y[i-1], x[i], y[i]))
    # Calculate curvature using the previous, current, and next points
    if i < len(x) - 1:
        x1, y1 = x[i-1], y[i-1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[i+1], y[i+1]
    else:
        x1, y1 = x[i-1], y[i-1]
        x2, y2 = x[i], y[i]
        x3, y3 = x[0], y[0]  # Close the loop
    a = euclidean_distance(x1, y1, x2, y2)
    b = euclidean_distance(x2, y2, x3, y3)
    c = euclidean_distance(x1, y1, x3, y3)
    curvature_value = 0 if a * b * c == 0 else 2 * np.abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / (a * b * c)
    curvature.append(curvature_value)
    psi = np.arctan2(y[i] - y[i-1], x[i] - x[i-1])
    psi_rad.append(psi)

# Close the loop for the last point
curvature.append(curvature[-1])
psi_rad.append(psi_rad[-1])

# Use pandas rolling to smooth the data while keeping the length unchanged
def smooth_data(data, window_size):
    return pd.Series(data).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

# Set the window size for smoothing (can be adjusted)
window_size = 11

# Smooth the curvature
curvature_smoothed = smooth_data(curvature, window_size)
# Smooth the heading angle (psi)
psi_rad_smoothed = median_filter(psi_rad, size=3)

# Update original curvature and psi_rad with smoothed data
curvature = curvature_smoothed
psi_rad = psi_rad_smoothed

# Step 1: Calculate initial velocities based on maximum lateral acceleration
for kappa in curvature:
    if kappa < 0.05:
        vx = max_velocity
    else:
        vx = np.sqrt(a_y_max / kappa) if kappa > 0 else max_velocity
    vx_mps.append(min(max(vx, min_velocity), max_velocity))

# Step 2: Forward propagation of longitudinal acceleration
for i in range(1, len(vx_mps)):
    d_s = s_m[i] - s_m[i-1]
    v_prev = vx_mps[i-1]
    if curvature[i-1] > 0.05:
        roti = (v_prev**2 * curvature[i-1] / a_y_max)**2
        if roti >=1:
            roti = 1
        ax = a_x_max * np.sqrt(1 - roti)
    else:
        ax = a_x_max
    ax_mps2[i-1] = ax
    v_new = np.sqrt(v_prev**2 + 2 * ax * d_s)
    vx_mps[i] = min(vx_mps[i], v_new)

# Step 3: Backward propagation of longitudinal acceleration
for i in range(len(vx_mps) - 2, -1, -1):
    d_s = s_m[i+1] - s_m[i]
    v_next = vx_mps[i+1]
    if curvature[i] > 0.05:
        roti = (v_prev**2 * curvature[i-1] / a_y_max)**2
        if roti >=1:
            roti = 1
        ax = a_x_max * np.sqrt(1 - roti)
    else:
        ax = a_x_max
    v_new = np.sqrt(max(v_next**2 + 2 * ax * d_s, 0)) 
    vx_mps[i] = min(vx_mps[i], v_new)
    ax_mps2[i-1] = vx_mps[i] - vx_mps[i-1]

# Add computed columns back to the dataframe
waypoints_df['s_m'] = s_m
waypoints_df['psi_rad'] = psi_rad
waypoints_df['kappa_radpm'] = curvature
waypoints_df['v_mps'] = vx_mps
waypoints_df['a_mps2'] = ax_mps2

# Reorder columns
column_order = ['s_m', 'x_m', 'y_m', 'psi_rad', 'kappa_radpm', 'v_mps', 'a_mps2']
waypoints_df = waypoints_df[column_order]

# Save results to a CSV file
output_path = f"{MAP_NAME}_calculation.csv"
waypoints_df.to_csv(output_path, index=False)

print(f"File saved successfully to {output_path}")