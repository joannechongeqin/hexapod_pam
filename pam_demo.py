from Yuna import Yuna
import time
import torch

GROUND_PLANE = 0.0 # height of ground plane
PLANE1 = 0.2
PLANE2 = 0.5

# leg_idxs = [0, 1]
# legs_on_ground = [True, True, True, True, True, True]
# pos = torch.tensor([[1.2, 0.3, PLANE2],
#                     [1.2, -0.3, PLANE2]])
# rot = torch.zeros_like(pos)

leg_idxs = [1]
legs_on_ground = [True, False, True, True, True, True]
pos = torch.tensor([[1.2, -0.3, PLANE2]])
rot = torch.zeros_like(pos)

yuna = Yuna(real_robot_control=0, pybullet_on=1, opt_vis=True, load_fyp_map=True, goal=pos.tolist())
yuna.env.camerafollow = False

# yuna.height_map.plot()

yuna.pam(pos, rot, leg_idxs, legs_on_ground)

time.sleep(10)

# NOTE: kinda working now but need fix the last leg raising up (singularity problem?)
