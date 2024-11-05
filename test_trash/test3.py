from Yuna import Yuna
import time
import numpy as np

yuna = Yuna(real_robot_control=0, pybullet_on=1)
yuna.env.camerafollow = False
yuna.env.load_wall()
time.sleep(3)
# yuna.env.add_y_ref_line_at_height(0.145)
print(yuna.env.get_elbow_pos())


front_x, front_y = 0.51589, 0.23145
middle_x, middle_y = 0.0575, 0.5125
back_x, back_y = 0.45839, 0.33105
front_y_diff = middle_y - front_y
back_y_diff = middle_y - back_y
raise_h = 0.3
raise_add_h = 0.15
wall_dist_from_origin = 0.6
leg1_wall_dist = wall_dist_from_origin - front_y
leg3_wall_dist = wall_dist_from_origin - middle_y
leg5_wall_dist = wall_dist_from_origin - back_y

front_back_elbow_x = 0.46
front_x_align = 0.25 # abs(front_x - front_back_elbow_x)
back_x_align = 0.3 # abs(back_x - front_back_elbow_x)

leg26raise = np.zeros((3,6))
leg26raise[:, 1] = [0, 0, raise_h]
leg26raise[:, 5] = [0, 0, raise_h]
leg26nearwall = np.zeros((3,6))
# leg26nearwall[:, 1] = [-front_x_align, -front_y_diff, -raise_h]
# leg26nearwall[:, 5] = [back_x_align, -back_y_diff, -raise_h]
leg26nearwall[:, 1] = [0.1, 0.2, -raise_h]
leg26nearwall[:, 5] = [-0.1, 0.2, -raise_h]
yuna.move_legs_by_pos_in_world_frame([leg26raise, leg26nearwall])

# leg4raise_w = np.zeros((3,6))
# leg4raise_w[:, 3] = [0, 0, 0.1]
# leg4wall_w = np.zeros((3,6))
# leg4wall_w[:, 3] = [0, 0.05, -0.1]
# yuna.move_legs_by_pos_in_world_frame([leg4raise_w, leg4wall_w])

# yuna.rotx_body(10, move=True)

leg15raise = np.zeros((3,6))
leg15raise[:, 0] = [0, 0, raise_h]
leg15raise[:, 4] = [0, 0, raise_h]
leg15wall = np.zeros((3,6))
leg15wall[:, 0] = [-front_x_align, leg1_wall_dist, 0.15]
leg15wall[:, 4] = [back_x_align, leg5_wall_dist, 0.15]
yuna.move_legs_by_pos_in_world_frame([leg15raise, leg15wall])

leg3raise = np.zeros((3,6))
leg3raise[:, 2] = [0, 0, raise_h]
leg3wall = np.zeros((3,6))
leg3wall[:, 2] = [0, leg3_wall_dist, raise_add_h]
yuna.move_legs_by_pos_in_world_frame([leg3raise, leg3wall])

time.sleep(10)

yuna.trans_body(0, 0.05, 0.05, move=True)
yuna.rotx_trans_body(10, 0, 0.02, 0.05, move=True)

# leg4raise_w = np.zeros((3,6))
# leg4raise_w[:, 3] = [0, 0, 0.1]
# leg4wall_w = np.zeros((3,6))
# leg4wall_w[:, 3] = [0, 0.05, -0.1]
# yuna.move_legs_by_pos_in_world_frame([leg4raise_w, leg4wall_w])

yuna.trans_body(0, 0.03, 0, move=True)

leg1away_w = np.zeros((3,6))
leg1away_w[:, 0] = [0, -0.1, 0]
leg1wall_w = np.zeros((3,6))
leg1wall_w[:, 0] = [0, 0.1, 0.2]
yuna.move_legs_by_pos_in_world_frame([leg1away_w, leg1wall_w])

leg3away_w = np.zeros((3,6))
leg3away_w[:, 2] = [0, -0.1, 0]
leg3wall_w = np.zeros((3,6))
leg3wall_w[:, 2] = [0, 0.1, 0.2]
yuna.move_legs_by_pos_in_world_frame([leg3away_w, leg3wall_w])

leg5away_w = np.zeros((3,6))
leg5away_w[:, 4] = [0, -0.1, 0]
leg5wall_w = np.zeros((3,6))
leg5wall_w[:, 4] = [0, 0.1, 0.2]
yuna.move_legs_by_pos_in_world_frame([leg5away_w, leg5wall_w])

yuna.rotx_trans_body(5, 0, 0, 0, move=True)
# yuna.trans_body(0.05, 0, 0, move=True)

# leg4raise_w = np.zeros((3,6))
# leg4raise_w[:, 3] = [0, 0, 0.1]
# leg4wall_w = np.zeros((3,6))
# leg4wall_w[:, 3] = [0.25, 0.05, -0.1]
# yuna.move_legs_by_pos_in_world_frame([leg4raise_w, leg4wall_w])

# leg6raise = np.zeros((3,6))
# leg6raise[:, 5] = [0, 0, 0.1]
# leg6nearwall = np.zeros((3,6))
# leg6nearwall[:, 5] = [0, 0.05, -0.1]
# yuna.move_legs_by_pos_in_world_frame([leg6raise, leg6nearwall])


leg2raise = np.zeros((3,6))
leg2raise[:, 1] = [0, 0, 0.1]
leg2nearwall = np.zeros((3,6))
leg2nearwall[:, 1] = [0, 0.05, 0]
yuna.move_legs_by_pos_in_world_frame([leg2raise])


# yuna.trans_body(0, 0.05, 0.05, move=True)

time.sleep(1e5)
