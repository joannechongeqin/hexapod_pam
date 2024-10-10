from Yuna import Yuna
import time
import numpy as np

yuna = Yuna(real_robot_control=0, pybullet_on=1)
yuna.env.camerafollow = False
yuna.env.load_wall()
yuna.env.add_y_ref_line_at_height(0.4)

front_x, front_y = 0.51589, 0.23145
middle_x, middle_y = 0.0575, 0.5125
back_x, back_y = 0.45839, 0.33105
front_y_diff = middle_y - front_y
back_y_diff = middle_y - back_y
raise_h = 0.2
raise_add_h = 0.15
wall_dist_from_origin = 0.6
leg1_wall_dist = wall_dist_from_origin - front_y
leg3_wall_dist = wall_dist_from_origin - middle_y - .01
leg5_wall_dist = wall_dist_from_origin - back_y
front_x_align = 0.25 
back_x_align = 0.3

leg16raise = np.zeros((3,6))
leg16raise[:, 0] = [-front_x_align/2, front_y_diff/2, raise_h]
leg16raise[:, 5] = [back_x_align/2, -back_y_diff/2, raise_h]
leg16align = np.zeros((3,6))
leg16align[:, 0] = [-front_x_align/2, front_y_diff/2, -raise_h]
leg16align[:, 5] = [back_x_align/2, -back_y_diff/2, -raise_h]
yuna.move_legs_by_pos_in_world_frame([leg16raise, leg16align])
print("aligned 16")

leg25raise = np.zeros((3,6))
leg25raise[:, 1] = [-front_x_align/2, -front_y_diff/2, raise_h]
leg25raise[:, 4] = [back_x_align/2, back_y_diff/2, raise_h]
leg25align = np.zeros((3,6))
leg25align[:, 1] = [-front_x_align/2, -front_y_diff/2, -raise_h]
leg25align[:, 4] = [back_x_align/2, back_y_diff/2, -raise_h]
yuna.move_legs_by_pos_in_world_frame([leg25raise, leg25align])
print("aligned 25")


N = 10 # total number of discretized  segments
# q0, x0 = yuna.env.get_body_pose() # initial body orientation (wrt x) and position in world frame
# q0 = q0[0]
# x0 = np.array(x0)
# print(f"q0: {q0}, x0: {x0}")
# xd = np.array([0, 1, 1]) # diff between initial and final body position in world frame
# qd = 30 * np.pi / 180  # diff between initial and final orientation in radians

qb_deg = 60 # diff between initial and target body orientation in degrees
xb = np.array([0, 0.2, 0.2]) # diff between initial and target body position in world frame
y_leg = 0.5 # diff in y between initial and final ground leg position in world frame (distance moved nearer to wall)
z_leg = 1.2 # diff in z between first wall step and final wall leg position in world frame (distance moved up)
dqb_deg = qb_deg/N
dxb = xb/N
dy_leg = y_leg/N
dz_leg = z_leg/N

# yuna.wall_transition_first_step_wall_leg(step_height=0.35, wall_dist=leg3_wall_dist)
# yuna.wall_transition_step_wall_leg(step_len=dz_leg)

for i in range(N):
    # LINEAR INTERPOLATION OF BOTH Y Z ROLL
    yuna.rotx_body(dqb_deg, move=True)
    print(f"rotated body {dqb_deg}deg")
    yuna.trans_body_in_world_frame(dxb[0], dxb[1], dxb[2], move=True)
    print(f"translated body {dxb}")
    if i <= 2 or i % 2 == 0:
        yuna.wall_transition_step_ground_leg(step_len=dy_leg, leg4_step_half=True)
    else:
        yuna.wall_transition_step_ground_leg(step_len=dy_leg, leg4_step_half=False)
    print(f"moved ground leg {dy_leg} closer to wall")
    if i == 0:
        continue
    if i == 1:
        yuna.wall_transition_first_step_wall_leg(step_height=0.46, wall_dist=leg3_wall_dist)
        print("transition to wall")
    else:
        yuna.wall_transition_step_wall_leg(step_len=dz_leg)
        print(f"moved wall leg {dz_leg} up")

for i in range(18):
    yuna.env.plot_reaction_forces_and_torque(i)
print("done plot")

# Lists to store orientation and position for each step
# qb_list = []
# xb_list = []
# for n in range(N+1): # n = 1, 2, 3, ..., N = current step
#     qb = q0 + n / N * qd # qf = target body orientation
#     xb = x0 + np.array([1-np.cos(qb), 1-np.cos(qb), np.sin(qb)] * xd) # xf = target body position
#     qb_list.append(qb)
#     xb_list.append(xb)
#     print(f"n: {n}, qb: {np.rad2deg(qb)}, xb: {xb}")

# yuna.env.add_body_ref_points_wrt_body_frame(xb_list, lifeTime=0)

# qb_prev, xb_prev = q0, x0
# for i in range(N+1):
#     qb, xb = qb_list[i], xb_list[i]
#     x_diff = xb - xb_prev
#     yuna.rotx_body(np.rad2deg(qb - qb_prev), move=True)
#     yuna.trans_body(x_diff[0], x_diff[1], x_diff[2], move=True)
#     qb_prev, xb_prev = qb, xb

time.sleep(1e5)
