from Yuna import Yuna
import time
import numpy as np

# try to raise leg2 when 3 legs on wall alr (but cannot support)

yuna = Yuna(real_robot_control=0, pybullet_on=1)
yuna.env.camerafollow = False
yuna.env.load_wall()
yuna.env.add_y_ref_line_at_height(0.145)

front_x, front_y = 0.51589, 0.23145
middle_x, middle_y = 0.0575, 0.5125
back_x, back_y = 0.45839, 0.33105
front_y_diff = middle_y - front_y
back_y_diff = middle_y - back_y
raise_h = 0.35
raise_add_h = 0.15
leg1_wall_dist = 0.6-0.23145
leg3_wall_dist = 0.6-0.5125
leg5_wall_dist = 0.6-0.33105

leg26raise = np.zeros((3,6))
leg26raise[:, 1] = [0, 0, raise_h]
leg26raise[:, 5] = [0, 0, raise_h]
leg26nearwall = np.zeros((3,6))
leg26nearwall[:, 1] = [0.1, 0.15, -raise_h]
leg26nearwall[:, 5] = [-0.1, 0.15, -raise_h]
yuna.move_legs_by_pos_in_world_frame([leg26raise, leg26nearwall])

# yuna.rotx_body(10, move=True)

leg15raise = np.zeros((3,6))
leg15raise[:, 0] = [0, 0, raise_h]
leg15raise[:, 4] = [0, 0, raise_h]
leg15wall = np.zeros((3,6))
leg15wall[:, 0] = [-front_y_diff, leg1_wall_dist, raise_add_h]
leg15wall[:, 4] = [back_y_diff, leg5_wall_dist, raise_add_h]
yuna.move_legs_by_pos_in_world_frame([leg15raise, leg15wall])

leg3raise = np.zeros((3,6))
leg3raise[:, 2] = [0, 0, raise_h]
leg3wall = np.zeros((3,6))
leg3wall[:, 2] = [0, leg3_wall_dist, raise_add_h]
yuna.move_legs_by_pos_in_world_frame([leg3raise, leg3wall])

yuna.trans_body(0, 0.05, 0.05, move=True)

leg4raise = np.zeros((3,6))
leg4raise[:, 3] = [0, 0, 0.1]
leg4nearwall = np.zeros((3,6))
leg4nearwall[:, 3] = [0, 0.05, -0.1]
yuna.move_legs_by_pos_in_world_frame([leg4raise, leg4nearwall])

leg2raise_w = np.zeros((3,6))
leg2raise_w[:, 1] = [0, 0, 0.1]
leg2nearwall = np.zeros((3,6))
leg2nearwall[:, 1] = [0, 0.05, -0.1]
yuna.move_legs_by_pos_in_world_frame([leg2raise_w, leg2nearwall])

# leg5raise_w = np.zeros((3,6))
# leg5raise_w[:, 4] = [0, 0, 0.2]
# leg5wall_w = np.zeros((3,6))
# leg5wall_w[:, 4] = [0, 0.07, 0.1]
# yuna.move_legs_by_pos_in_world_frame([leg5raise_w, leg5wall_w])

# leg3raise_w = np.zeros((3,6))
# leg3raise_w[:, 2] = [0, 0, 0.2]
# leg3wall_w = np.zeros((3,6))
# leg3wall_w[:, 2] = [0, 0.07, 0.1]
# yuna.move_legs_by_pos_in_world_frame([leg3raise_w, leg3wall_w])

# yuna.rotx_trans_body(-15, 0, 0.05, 0.05, move=True)

time.sleep(1e5)
