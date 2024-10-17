from Yuna import Yuna
import time
import numpy as np

yuna = Yuna(real_robot_control=0, pybullet_on=1, show_ref_points=1)
yuna.env.camerafollow = False
yuna.env.load_wall()

yuna.env.add_y_ref_line_at_height(0.145+0.2)
# print("legPos: ", yuna.env.get_leg_pos())
# print("elbowPos: ", yuna.env.get_elbow_pos())
# time.sleep(1e5)

front_x, front_y = 0.51589, 0.23145
middle_x, middle_y = 0.0575, 0.5125
back_x, back_y = 0.45839, 0.33105
front_y_diff = middle_y - front_y
back_y_diff = middle_y - back_y
raise_h = 0.35
raise_add_h = 0.15
wall_dist_from_origin = 0.6
leg1_wall_dist = wall_dist_from_origin - front_y
leg3_wall_dist = wall_dist_from_origin - middle_y
leg5_wall_dist = wall_dist_from_origin - back_y

leg26raise = np.zeros((3,6))
leg26raise[:, 1] = [0.05, 0.075, raise_h]
leg26raise[:, 5] = [-0.05, 0.075, raise_h]
leg26nearwall = np.zeros((3,6))
leg26nearwall[:, 1] = [0.05, 0.075, -raise_h]
leg26nearwall[:, 5] = [-0.05, 0.075, -raise_h]
yuna.move_legs_by_pos_in_world_frame([leg26raise, leg26nearwall])
print("moved leg 26 near wall")

leg4raise_w = np.zeros((3,6))
leg4raise_w[:, 3] = [0, 0.025, 0.1]
leg4wall_w = np.zeros((3,6))
leg4wall_w[:, 3] = [0, 0.025, -0.1]
yuna.move_legs_by_pos_in_world_frame([leg4raise_w, leg4wall_w])
print("moved leg 4 near wall")

yuna.rotx_body(6, move=True)
yuna.trans_body(0, 0.02, 0.02, move=True)
print("translated body and rotated body")

leg15raise = np.zeros((3,6))
leg15raise[:, 0] = [0, 0, raise_h]
leg15raise[:, 4] = [0, 0, raise_h]
leg15wall = np.zeros((3,6))
leg15wall[:, 0] = [-front_y_diff, leg1_wall_dist, 0.15]
leg15wall[:, 4] = [back_y_diff, leg5_wall_dist, 0.15]
yuna.move_legs_by_pos_in_world_frame([leg15raise, leg15wall])
print("moved leg 15 on wall")

leg3raise = np.zeros((3,6))
leg3raise[:, 2] = [0, 0, raise_h]
leg3wall = np.zeros((3,6))
leg3wall[:, 2] = [0, leg3_wall_dist, raise_add_h]
yuna.move_legs_by_pos_in_world_frame([leg3raise, leg3wall])
print("moved leg 3 on wall")

# yuna.rotx_body(6, move=True)
# yuna.trans_body(0, 0.02, 0.02, move=True)
# print("translated body and rotated body") 

N = 20
dqb_deg = 50 / N
dxb = np.array([0, 0.2, 0.2]) / N
dy_leg = 0.2 / N
dz_leg = 1.2 / N

for i in range(N):
    # LINEAR INTERPOLATION OF BOTH Y Z ROLL
    yuna.trans_body_in_world_frame(dxb[0], dxb[1], dxb[2], move=True)
    print(f"translated body {dxb}")
    yuna.rotx_body(dqb_deg, num_of_waypoints=1, move=True)
    print(f"rotated body {dqb_deg}deg")
    yuna.wall_transition_step_ground_leg(step_len=dy_leg)
    print(f"moved ground leg {dy_leg} closer to wall")
    yuna.wall_transition_step_wall_leg(step_len=dz_leg)
    print(f"moved wall leg {dz_leg} up")

# leg1away_w = np.zeros((3,6))
# leg1away_w[:, 0] = [0, -0.1, 0]
# leg1wall_w = np.zeros((3,6))
# leg1wall_w[:, 0] = [0, 0.1, 0.2]
# yuna.move_legs_by_pos_in_world_frame([leg1away_w, leg1wall_w])
# print("moved leg 1 higher on wall")

# leg3away_w = np.zeros((3,6))
# leg3away_w[:, 2] = [0, -0.1, 0]
# leg3wall_w = np.zeros((3,6))
# leg3wall_w[:, 2] = [0, 0.1, 0.2]
# yuna.move_legs_by_pos_in_world_frame([leg3away_w, leg3wall_w])
# print("moved leg 3 higher on wall")

# leg5away_w = np.zeros((3,6))
# leg5away_w[:, 4] = [0, -0.1, 0]
# leg5wall_w = np.zeros((3,6))
# leg5wall_w[:, 4] = [0, 0.1, 0.2]
# yuna.move_legs_by_pos_in_world_frame([leg5away_w, leg5wall_w])
# print("moved leg 5 higher on wall")

# yuna.rotx_body(5, move=True)
# print("rotated body 5 deg")

# leg4raise_w = np.zeros((3,6))
# leg4raise_w[:, 3] = [0.05, 0.02, .1]
# leg4wall_w = np.zeros((3,6))
# leg4wall_w[:, 3] = [0.2, 0.03, -0.1]
# yuna.move_legs_by_pos_in_world_frame([leg4raise_w, leg4wall_w])
# print("moved leg 4 near leg 2")

# leg6raise = np.zeros((3,6))
# leg6raise[:, 5] = [0, 0, 0.1]
# leg6forward = np.zeros((3,6))
# leg6forward[:, 5] = [0.1, 0, -0.1]
# yuna.move_legs_by_pos_in_world_frame([leg6raise, leg6forward])

# yuna.trans_body(0.05, 0, 0, move=True)

# leg5away_w = np.zeros((3,6))
# leg5away_w[:, 4] = [0, -0.1, 0]
# leg5wall_w = np.zeros((3,6))
# leg5wall_w[:, 4] = [0, 0.1, 0.2]
# yuna.move_legs_by_pos_in_world_frame([leg5away_w, leg5wall_w])

# leg1away_w = np.zeros((3,6))
# leg1away_w[:, 0] = [0.05, -0.1, 0]
# leg1wall_w = np.zeros((3,6))
# leg1wall_w[:, 0] = [0.1, 0.1, 0.2]
# yuna.move_legs_by_pos_in_world_frame([leg1away_w, leg1wall_w])

# leg3away_w = np.zeros((3,6))
# leg3away_w[:, 2] = [0, -0.1, 0]
# leg3wall_w = np.zeros((3,6))
# leg3wall_w[:, 2] = [0, 0.1, 0.2]
# yuna.move_legs_by_pos_in_world_frame([leg3away_w, leg3wall_w])

# leg4raise_w = np.zeros((3,6))
# leg4raise_w[:, 3] = [0, 0, 0.1]
# leg4wall_w = np.zeros((3,6))
# leg4wall_w[:, 3] = [0, 0.05, -0.1]
# yuna.move_legs_by_pos_in_world_frame([leg4raise_w, leg4wall_w])

# yuna.trans_body(0, 0.03, 0, move=True)

# leg4raise_w = np.zeros((3,6))
# leg4raise_w[:, 3] = [-0.05, 0.1, 0.1]
# leg4forward_w = np.zeros((3,6))
# leg4forward_w[:, 3] = [-0.25, 0.05, -0.1]
# yuna.move_legs_by_pos_in_world_frame([leg4raise_w, leg4forward_w])

# leg2raise = np.zeros((3,6))
# leg2raise[:, 1] = [0, 0, 0.1]
# leg2nearwall = np.zeros((3,6))
# leg2nearwall[:, 1] = [0, 0.05, 0]
# yuna.move_legs_by_pos_in_world_frame([leg2raise])


# yuna.trans_body(0, 0.05, 0.05, move=True)

for i in range(18):
    print(i)
    yuna.env.plot_reaction_forces_and_torque(i)

time.sleep(1e5)
