import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

q0 = 0 # initial orientation
N = 20 # total number of discretized  segments
qd = 30 * np.pi / 180  # diff between initial and final orientation in radians
x0 = np.array([0, 0, 0.12]) # initial body position in world frame
xd = np.array([0, 1, 1]) # diff between initial and final body position in world frame

# Lists to store orientation and position for each step
qb_list = []
xb_list = []

for n in range(N+1): # n = 1, 2, 3, ..., N = current step
    qb = q0 + n / N * qd # qf = target body orientation
    xb = x0 + np.array([1-np.cos(qb), 1-np.cos(qb), np.sin(qb)]) * xd / N  # xf = target body position
    qb_list.append(qb)
    xb_list.append(xb)
    
# Convert lists to arrays for easier plotting
qb_array = np.array(qb_list)
xb_array = np.array(xb_list)

# Plotting qb (orientation) vs steps
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(N+1), qb_array, 'r-', label='Orientation (qb)')
plt.xlabel('Step (n)')
plt.ylabel('Orientation (qb) [radians]')
plt.title('Orientation (qb) vs Steps')
plt.grid(True)
plt.legend()

# 3D plot for xb
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the xb positions in 3D space
ax.plot(xb_array[:, 0], xb_array[:, 1], xb_array[:, 2], 'bo-', label='Position (xb)')
ax.scatter(xb_array[:, 0], xb_array[:, 1], xb_array[:, 2], c='r', marker='o')  # Mark each step

# Labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Plot of Position (xb)')
ax.grid(True)
ax.legend()

plt.show()