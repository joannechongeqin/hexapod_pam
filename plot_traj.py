
'''
plot the trajectory of the Yuna's 2 sets of tripod
'''
from Yuna_TrajPlanner import TrajPlanner
import numpy as np
import matplotlib.pyplot as plt

tp = TrajPlanner()

tripod1 = [0 ,3, 4]
tripod2 = [1, 2, 5]

step_len = 0.2
course = - np.pi / 6
rotation = np.pi / 9
init_pose =np.zeros((4,6))

traj = np.zeros((tp.traj_dim, 3, 6))
for timestep in range(tp.traj_dim):
    for leg in range(6):
        traj[timestep], end_pose = tp.get_loco_traj(init_pose=init_pose, step_len=step_len, course=course, rotation=rotation, flag=0, timestep=timestep)
fig, ax = plt.subplots()
t1, t2 = [], []
flag = 0
for step in range(tp.traj_dim):
    if step % 3 == 0 or step + 1 == tp.traj_dim:
        triangle1, triangle2 = [], []
        for i in tripod1:
            triangle1.append([traj[step,0,i],traj[step,1,i]])
        triangle1.append([traj[step,0,tripod1[0]],traj[step,1,tripod1[0]]])
        for i in tripod2:
            triangle2.append([traj[step,0,i],traj[step,1,i]])
        triangle2.append([traj[step,0,tripod2[0]],traj[step,1,tripod2[0]]])
        t1.append(np.array(triangle1))
        t2.append(np.array(triangle2))
        ax.fill(t1[flag][:,0], -t1[flag][:,1], color=(1,0,0,0.01))
        ax.fill(t2[flag][:,0], -t2[flag][:,1], color=(0,0,1,0.01))
        flag += 1

ax.plot(np.array(t1)[0,:,0], -np.array(t1)[0,:,1], color=(1,0,0,1), linestyle='-.', label='Swing Tripod Initial Position')
ax.plot(np.array(t2)[0,:,0], -np.array(t2)[0,:,1], color=(0,0,1,1), linestyle='-.', label='Stance Tripod Initial Position')
ax.plot(np.array(t1)[-1,:,0], -np.array(t1)[-1,:,1], color=(1,0,0,0.2), label='Swing Tripod End Position')
ax.plot(np.array(t2)[-1,:,0], -np.array(t2)[-1,:,1], color=(0,0,1,0.2), label='Stance Tripod End Position')
for leg in tripod1:
    ax.plot(traj[:,0,leg], -traj[:,1,leg], color=(1,0,0,1), linestyle=':')
ax.plot([init_pose[0,leg],end_pose[0,leg]], [-init_pose[1,leg],-end_pose[1,leg]], color=(1,0,0,1), linestyle=':', label='Swing Tripod and Legs Trajectories')
for leg in tripod2:
    ax.plot(traj[:,0,leg], -traj[:,1,leg], color=(0,0,1,1), linestyle=':')
ax.plot([init_pose[0,leg],end_pose[0,leg]], [-init_pose[1,leg],-end_pose[1,leg]], color=(0,0,1,1), linestyle=':', label='Stance Tripod and Legs Trajectories')

ax.set_aspect('equal', 'box')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()