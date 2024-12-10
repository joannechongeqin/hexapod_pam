from pytorch_optimizer import *
from Yuna import Yuna
import time

# --- SET 1 ---
# leg_idxs = [0, 1]
# legs_on_ground = [False, False, True, True, True, True]
# legs_plane = [PLANE2, PLANE2, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE]
# pos = torch.tensor([[0.51589, 0.23145, PLANE2],
#                     [0.51589, -0.23145, PLANE2]])

# --- SET 2 ---
# leg_idxs = [2, 3]
# legs_on_ground = [True, True, False, False, True, True]
# legs_plane = [GROUND_PLANE, GROUND_PLANE, PLANE2, PLANE2, GROUND_PLANE, GROUND_PLANE]
# pos = torch.tensor([[0.0575, 0.5125, PLANE2],
#                     [0.0575, -0.5125, PLANE2]])

# --- SET 3 ---
leg_idxs = [0, 1]
legs_on_ground = [False, False, True, True, True, True]
legs_plane = [PLANE2, PLANE2, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE]
pos = torch.tensor([[0.6, 0.3, PLANE2],
                    [0.7, -0.2, PLANE2]])

rot = torch.zeros_like(pos)
goal = pk.Transform3d(pos=pos, rot=rot)
params = solve_multiple_legs_ik(goal, legs_on_ground=legs_on_ground, legs_plane=legs_plane, leg_idxs=leg_idxs, batch_size=batch_size)
robot_frame_trans_w, base_trans_w, leg_trans_w, leg_trans_r = get_transformations_from_params(params)

# visualize(base_trans=base_trans_w, leg_trans=leg_trans_w, goal=pos)

batch_idx = 0
final_eef_pos_r = leg_trans_r[batch_idx, :, -1, :3, 3].numpy().T
final_body_pos_w = base_trans_w[batch_idx, :3, 3].numpy()
final_eef_pos_w = leg_trans_w[batch_idx, :, -1, :3, 3].numpy().T
# print("base_trans_w: ", base_trans_w)
# # print("leg_trans_r: ", leg_trans_r[0])

# DIRECTLY SHOW OPTIMIZED EEF POS IN PYBULLET 
# NOTE: ROBOT BODY TILTED, MIGHT BE DUE TO HOW STEP() IS CALLED IN INIT_ROBOT -> TODO: TRY WRITING ROUTINE FIRST
# yuna = Yuna(real_robot_control=0, pybullet_on=1, eePos=final_eef_pos_r, bodyPos=final_body_pos_w)

# SHOW INITIAL POSITION IN PYBULLET
yuna = Yuna(real_robot_control=0, pybullet_on=1)
print("initial_body_pos: ", yuna.env.get_body_pose()[0])
print("initial_eef_pos_r: ", yuna.env.get_leg_pos())
print("initial_eef_pos_w: ", yuna.env.get_leg_pos_in_world_frame())
print()

# WRITE A ROUTINE TO MOVE ROBOT FROM INITIAL POSITION TO THE OPTIMIZED POSITION
yuna.trans_body_to_in_world_frame(np.array(final_body_pos_w), move=True)
yuna.swing_leg(4, final_eef_pos_w[:, 4]) # back leg
yuna.swing_leg(5, final_eef_pos_w[:, 5]) # back leg
yuna.swing_leg(2, final_eef_pos_w[:, 2]) # middle leg
yuna.swing_leg(3, final_eef_pos_w[:, 3]) # middle leg
yuna.swing_leg(0, final_eef_pos_w[:, 0]) # front leg
yuna.swing_leg(1, final_eef_pos_w[:, 1]) # front leg

print("target_body_pos: ", final_body_pos_w)
print("target_eef_pos_r: ", final_eef_pos_r)
print("target_eef_pos_w: ",final_eef_pos_w)
print()

print("final_body_pos: ", yuna.env.get_body_pose()[0])
print("final_eef_pos_r: ", yuna.env.get_leg_pos())
print("final_eef_pos_w: ", yuna.env.get_leg_pos_in_world_frame())


print("--- ERROR: ---")
print("body_pos_error: ", final_body_pos_w - yuna.env.get_body_pose()[0])
print("eef_pos_w_error: ", final_eef_pos_w - yuna.env.get_leg_pos_in_world_frame())

time.sleep(30)
