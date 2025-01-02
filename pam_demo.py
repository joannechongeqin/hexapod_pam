from Yuna import Yuna
import time
import torch

GROUND_PLANE = 0.0 # height of ground plane
PLANE1 = 0.2
PLANE2 = 0.5

# leg_idxs = [0, 1]
# legs_on_ground = [True, True, True, True, True, True]
# legs_plane = [PLANE2, PLANE2, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE]
# pos = torch.tensor([[1.2, 0.3, PLANE2],
#                     [1.2, -0.3, PLANE2]])
# rot = torch.zeros_like(pos)

leg_idxs = [0]
legs_on_ground = [False, True, True, True, True, True]
legs_plane = [0, PLANE2, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE]
pos = torch.tensor([[1.2, -0.3, PLANE2]])
rot = torch.zeros_like(pos)

yuna = Yuna(real_robot_control=0, pybullet_on=1,  opt_vis=False, goal=pos.tolist())
yuna.env.camerafollow = False

# yuna.height_map.plot()

yuna.pam(pos, rot, leg_idxs, legs_on_ground, legs_plane)

time.sleep(10)

# NOTE: kinda working now but need fix the last leg raising up (singularity problem?)
