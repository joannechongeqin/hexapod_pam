from Yuna import Yuna
import time
import torch

GROUND_PLANE = 0.0 # height of ground plane
PLANE1 = 0.2
PLANE2 = 0.35

leg_idxs = [0, 1]
legs_on_ground = [False, False, True, True, True, True]
legs_plane = [PLANE2, PLANE2, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE]
pos = torch.tensor([[1., 0.3, PLANE2],
                    [1., -0.2, PLANE2]])
rot = torch.zeros_like(pos)

yuna = Yuna(real_robot_control=0, pybullet_on=1,  opt_vis=False)
yuna.env.camerafollow = False

# yuna.height_map.plot()

yuna.pam(pos, rot, leg_idxs, legs_on_ground, legs_plane)

