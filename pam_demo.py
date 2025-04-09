from Yuna import Yuna
import time
import torch
import numpy as np

yuna = Yuna(real_robot_control=0)
yuna.env.camerafollow = False

# --- PRESS BUTTON WITH ONE LEG ---
# leg_idxs = [1]
# pos = torch.tensor([[1.45, -0.3, 0.7]])
# yuna.pam_press_button(pos, 1)

# -- OPTIMIZATION FOR RAISING TWO FRONT LEGS ---
# leg_idxs = [0, 1]
# pos = torch.tensor([[1.4, 0.3, 0.6], [1.4, -0.3, 0.6]])
# yuna.pam(pos, leg_idxs)


# --- DATA TESTED ON REAL ROBOT ---
# target [1.45, -0.3, 0.7], leg_idx 1
body_waypoints = np.array([
    [0.1434, -0.0003, 0.2016],
    [0.2828, -0.0019, 0.2681],
    [0.4254, -0.0018, 0.3217],
    [0.5652, -0.0043, 0.3340],
    [0.7045, -0.0033, 0.3571],
    [0.8477, -0.0061, 0.4140]
], dtype=np.float32)

legs_waypoints = np.array([
    [[0.6115, 0.6056, 0.1351, 0.1395, -0.3237, -0.3230],
     [0.3010, -0.3085, 0.5078, -0.5077, 0.2996, -0.3015],
     [0.0274, 0.0270, 0.0210, 0.0210, 0.0289, 0.0294]],
    [[0.7367, 0.7304, 0.3606, 0.3656, -0.1708, -0.1664],
     [0.2254, -0.2323, 0.4987, -0.5021, 0.3225, -0.3315],
     [0.2183, 0.2265, 0.0261, 0.0260, 0.0265, 0.0262]],
    [[0.8913, 0.8834, 0.4664, 0.4656, -0.0342, -0.0234],
     [0.2539, -0.2745, 0.5022, -0.5014, 0.3044, -0.3178],
     [0.2284, 0.2251, 0.0316, 0.0276, 0.0286, 0.0262]],
    [[1.0398, 1.0125, 0.5930, 0.6079, 0.1076, 0.1187],
     [0.2548, -0.2985, 0.4933, -0.5006, 0.3003, -0.3247],
     [0.2213, 0.2309, 0.0257, 0.0211, 0.0321, 0.0304]],
    [[1.1453, 1.1517, 0.7611, 0.7674, 0.2618, 0.2487],
     [0.2291, -0.2259, 0.4871, -0.4965, 0.3069, -0.2997],
     [0.3227, 0.3207, 0.2256, 0.2228, 0.0279, 0.0295]],
    [[1.3390, 1.4504, 0.8909, 1.0356, 0.4073, 0.4082],
     [0.0776, -0.2988, 0.4981, -0.4554, 0.2686, -0.2879],
     [0.3351, 0.7004, 0.2280, 0.2245, 0.0331, 0.0355]]
], dtype=np.float32)

# yuna = Yuna(real_robot_control=1, zed_on=True) # comment out this line to test on real robot
yuna.pam_move(body_waypoints, legs_waypoints, ("press_button", 1))

# retract leg after pressing button
retract_pos =  [np.zeros((3,6))]
retract_pos[0][0, 1] = -0.15
retract_pos[0][2, 1] = -0.2
yuna.move_legs_by_pos_in_world_frame(retract_pos)

time.sleep(100)