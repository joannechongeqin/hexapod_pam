from pytorch_optimizer import *
from Yuna import Yuna
import time

# --- SET 1 ---
# TODO: ROBOT CANT SUPPORT ITSELF
# leg_idxs = [0, 1]
# legs_on_ground = [False, False, True, True, True, True]
# legs_plane = [PLANE2, PLANE2, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE]
# pos = torch.tensor([[0.51589, 0.23145, PLANE1],
#                     [0.51589, -0.23145, PLANE1]])

# --- SET 2 ---
leg_idxs = [2, 3]
legs_on_ground = [True, True, False, False, True, True]
legs_plane = [GROUND_PLANE, GROUND_PLANE, PLANE2, PLANE2, GROUND_PLANE, GROUND_PLANE]
pos = torch.tensor([[0.0575, 0.5125, PLANE2],
                    [0.0575, -0.5125, PLANE2]])

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
print("final_body_pos: ", final_body_pos_w)
# # print("leg_trans_r: ", leg_trans_r[0])
print("final_eef_pos_r: ", final_eef_pos_r)
print("final_eef_pos_w: ",final_eef_pos_w)

yuna = Yuna(real_robot_control=0, pybullet_on=1, eePos=final_eef_pos_r, bodyPos=final_body_pos_w)

print("yuna body: ", yuna.env.get_body_matrix())
print("yuna eef_body: ", yuna.env.get_leg_pos())
print("yuna eef_world: ", yuna.env.get_leg_pos_in_world_frame())

time.sleep(10)
