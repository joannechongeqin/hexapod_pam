from Yuna_TrajPlanner import TrajPlanner
import numpy as np
import matplotlib.pyplot as plt

tp = TrajPlanner()
ip = np.array([0,0,0])
ep = np.array([1,0,0])
traj = tp.walk_swing_traj(ip, ep)
x = traj[:, 0]
y = traj[:, 2]
fig, ax = plt.subplots()
ax.scatter(x, y, s=1)
plt.show()