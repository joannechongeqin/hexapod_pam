import os
import math
import pytorch_kinematics as pk
import torch
import trimesh
from matplotlib import colormaps
import matplotlib.pyplot as plt
from torchmin import minimize
import numpy as np
from shapely.geometry import Point, Polygon

PI = math.pi
NUM_LEGS = 6
MAX_OPP_LEGS_DIST = (300 + 325 + 187.5) * 2 / 1000  # max distance between opposite legs, when it is fully stretched (in meters)
batch_size = 2

GROUND_PLANE = 0.0 # height of ground plane
PLANE1 = 0.1
PLANE2 = 0.2

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)
curr_dir = os.path.dirname(os.path.abspath(__file__))

chains = [] # list to store chain for each leg
# initial [guess] joint angles for robot's legs (base at origi, joints at 90 deg)
# 1x18, each group of 3 values represents joint angles for each leg
th_0 = torch.tensor([0, 0, -PI / 2, 
                     0, 0, PI / 2, 
                     0, 0, -PI / 2,
                     0, 0, PI / 2, 
                     0, 0, -PI / 2, 
                     0, 0, PI / 2]*batch_size).reshape((batch_size, 18))

    
def get_transformation_fk(theta, batch_size=batch_size):
    """
    Computes transformation matrices for each leg based on given joint angles.
    
    :params torch.Tensor theta: Joint angles of all legs. Shape: [batch_size, 18].
    :params int, optional batch_size: Number of samples processed at a time during a pass. 
    
    :return: base_trans, leg_trans
    - base_trans (torch.Tensor): Transformation matrix of body/base in robot frame. Shape: [batch_size, 4, 4].
    - leg_trans (torch.Tensor): Transformation matrices of each leg in robot frame. Shape: [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4].
    """
    leg_trans_r = []
    base_trans_r = []
    
    for i in range(NUM_LEGS):
        # build serial chain for each leg, add to the list of chains
        chain = pk.build_serial_chain_from_urdf(open(os.path.join(curr_dir, "urdf", "yuna.urdf")).read().encode(),f"leg{i+1}__FOOT") 
        # ^NOTE: suppressed "unknown tag warning" at /home/jceqin/.local/lib/python3.10/site-packages/pytorch_kinematics/urdf_parser_py/xml_reflection/core.py
        chains.append(chain)
        
        # extract joint angles for each leg
        theta_i = theta[:, 3*i:3*(i+1)]
        
        # perform FK for each leg, ret contains trans mat of all links of that leg
        ret = chain.forward_kinematics(theta_i, end_only=False)
        
        # extract trans mat for base link (should be same for all legs)
        if i == 0:
            base_trans_r = ret['base_link'].get_matrix()
        
        # extract trans mat for individual joints/links of the leg
        mat_base = ret[f'base{i+1}__INPUT_INTERFACE'].get_matrix()
        mat_shoulder = ret[f'shoulder{i+1}__INPUT_INTERFACE'].get_matrix()
        mat_elbow = ret[f'elbow{i+1}__INPUT_INTERFACE'].get_matrix()
        mat_leg = ret[f'leg{i+1}__LAST_LINK'].get_matrix()
        mat_foot = ret[f'leg{i+1}__FOOT'].get_matrix()
        
        # concatenate all the trans mat for the current leg
        trans_i = torch.cat([mat_base, mat_shoulder, mat_elbow, mat_leg, mat_foot])
        
        # reshape trans_i to [batch_size, num_of_links, 4, 4]  
        #   first reshape to dim [num_of_links, batch_size, 4, 4], then swap first two dim
        #   -1 -> auto infer size of that dim based on other dims and total num of elems in tensor
        # then append to leg_trans list      
        leg_trans_r.append(trans_i.reshape(-1, batch_size, 4, 4).transpose(0, 1))
        # print(f"base_{i}: \n", base_trans_r)
        # print(f"trans_{i}: \n", trans_i)
    
    # reshape to dim [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4]
    leg_trans_r = torch.cat(leg_trans_r).reshape(NUM_LEGS, batch_size, -1, 4, 4).transpose(0, 1)
    return base_trans_r, leg_trans_r


def get_transformations_from_params(params):
    """
    Computes base and leg transformation matrices from the given parameters.

    :param torch.Tensor params: A tensor containing joint angles and base position.
                                Shape: [batch_size, 21], where the first 18 values are joint angles, 
                                and the remaining 3 are base parameters (x,y,z) in world frame. 

    :return: robot_frame_trans, base_trans, leg_trans
    - robot_frame_trans (torch.Tensor): Transformation matrix of robot frame in world frame. Shape: [batch_size, 4, 4].
    - base_trans (torch.Tensor): Base transformation matrix in world frame. Shape: [batch_size, 4, 4].
    - leg_trans (torch.Tensor): Leg transformation matrices in world frame. Shape: [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4].
    """
    rob_tf_w = pk.Transform3d(pos=torch.cat([params[:,18:]],dim=-1), 
                            rot=torch.cat([torch.zeros(batch_size,3)],dim=-1))
    robot_frame_trans_w = rob_tf_w.get_matrix() # in world frame
    base_trans_r, leg_trans_r = get_transformation_fk(params[:,:18]) # in robot frame

    # transform to world frame
    base_trans_w = torch.bmm(robot_frame_trans_w,base_trans_r)
    leg_trans_w = torch.einsum('bkl,bijlm->bijkm',robot_frame_trans_w,leg_trans_r) # einsum = Einstein summation notation, for specifying complex tensor operations with concise notation
    
    return robot_frame_trans_w, base_trans_w, leg_trans_w


def check_pose_validity(leg_pos, body_pos, legs_on_ground):
    """
    :param torch.Tensor leg_pos: Positions of the end-effectors of all legs. Shape: [NUM_LEGS, 3].
    :param torch.Tensor body_pos: Position of the body CoM. Shape: [3].
    """
    # print("check_pose_validity leg_pos: ", leg_pos)
    # print("check_pose_validity body_pos: ", body_pos)
    # --- if supported by less than 3 legs, robot is confirm not stable ---
    if sum(legs_on_ground) < 3:
        print("Invalid pose: Less than 3 legs on ground, robot is not stable")
        return False

    # --- check if distance between opposite legs is achievable ---
    def get_distance(p1, p2):
        return torch.norm(p1 - p2).item()
    
    opp_pairs = [(0, 5), (1, 4), (2, 3)]
    for i, j in opp_pairs:
        # print(f"Distance between legs {i} and {j}: {get_distance(goal[i], goal[j])}")
        if get_distance(leg_pos[i], leg_pos[j]) > MAX_OPP_LEGS_DIST:
            print(f"Invalid pose: Distance between legs {i} and {j} (opposite pair) is too far")
            return False
    
    # --- check if body CoM is within support polygon ---
    support_idxs = [i for i in range(NUM_LEGS) if legs_on_ground[i]]
    support_polygon = Polygon(leg_pos[support_idxs, :2].detach().numpy()).convex_hull
    body_xy = Point(body_pos[:2].detach().numpy())

    # # visualization
    # fig, ax = plt.subplots()
    # centroid = support_polygon.centroid
    # x, y = support_polygon.exterior.xy
    # ax.fill(x, y, alpha=0.5, fc='blue', ec='black', label='support polygon')
    # ax.plot(body_xy.x, body_xy.y, 'ro', label='body')
    # ax.plot(centroid.x, centroid.y, 'go', label='centroid')
    # ax.legend()
    # ax.grid(True)
    # plt.show()

    if not support_polygon.contains(body_xy):
        print("Invalid pose: Body CoM is outside of support polygon")
        return False
    
    return True



def solve_multiple_legs_ik(goal, leg_idxs, legs_on_ground, legs_plane, batch_size=1):
    """
    Solves the inverse kinematics for multiple legs, optimizing to reach specified goal positions.

    :param pk.Transform3d goal: The desired position and orientation for the leg end-effectors. Shape: [len(leg_idxs), 4, 4].
    :param list leg_idxs: List of leg indices to be optimized.
    :param list legs_on_ground: List of boolean values indicating whether each leg is on the ground.
    :param list legs_plane: List of heights of the planes on which the legs are placed.
    :param int batch_size: The number of samples processed at a time during a pass.

    :return: optimized_params
    - optimized_params (torch.Tensor): A tensor containing optimized joint angles and base position.
                                Shape: [batch_size, 21], where the first 18 values are joint angles, 
                                and the remaining 3 are base parameters (x,y,z) in world frame. 
    """
    assert goal.get_matrix().shape[0] == len(leg_idxs)
    
    # generate random initial guess for optimization
    theta = th_0 # torch.rand(batch_size, 18) # joint angles
    base_xyz_w = torch.rand(batch_size, 3) # xyz base position
    # z_rot = torch.rand(batch_size, 1) # z rotation angle of base
    params = torch.cat([theta,base_xyz_w], dim=-1)
    
    def cost_function(params):
        # --- optimize based on distance between eef pos and goal pos ---
        #       rob_tf = Transform3d(pos=tensor([[x, y, z]]), rot=tensor([[quaternion (w,x,y,z) of euler angle (0, 0, z_rot)]])) 
        #       torch.cat(..., dim=-1) concatenates input tensors along last dim
        #       .unsqueeze(dim) adds a new dim of size 1 at the specified position (for proper concatenation)
        rob_tf_w = pk.Transform3d(pos=torch.cat([params[:,18:]],dim=-1), 
                                    rot=torch.cat([torch.zeros(batch_size,3)],dim=-1))
        base_trans_r, leg_trans_r = get_transformation_fk(params[:,:18]) # [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4]
        eef_trans_r = leg_trans_r[:,:,-1,:,:] # [batch_size, NUM_Legs, 4, 4] (select last elem along 3rd dim)

        all_eef_pos_w = torch.einsum("bkl,bilm->bikm",rob_tf_w.get_matrix(),eef_trans_r)
        target_eef_pos_w = all_eef_pos_w[:,leg_idxs,:3,3] # extract pos (xyz) of end-effectors
        # print(f"all_eef_pos: ", all_eef_pos)
        # print("target_eef_pos: ", target_eef_pos)
        eef_pos_residual_squared = (target_eef_pos_w - goal.get_matrix()[:,:3,3].unsqueeze(0))**2 # calculate squared diff between eef pos and goal pos

        # --- optimize based on free legs' height ---
        free_legs = [i for i in range(NUM_LEGS) if i not in leg_idxs]
        free_legs_height = all_eef_pos_w[:, free_legs, 2, 3]
        free_legs_plane = torch.tensor(legs_plane)[free_legs].repeat(batch_size, 1)
        # print("free_legs_height: ", free_legs_height)
        # print("free_legs_plane: ", free_legs_plane)
        free_legs_residual_squared = (free_legs_height - free_legs_plane) ** 2
        
        # --- optimize based on static stability margin (min dist from the CoM to the edges of support polygon) ---
        # project all points to a ground plane and draw support polygon using legs on ground
        # then minimize distance of body CoM to the centroid of the support polygon
        eef_support_idxs = [i for i in range(NUM_LEGS) if legs_on_ground[i]]
        # print("eef_support_idxs: ", eef_support_idxs)
        eef_support_xy_pos = all_eef_pos_w[:, eef_support_idxs, :2, 3]
        print(eef_support_idxs)

        # print("eef_support_xy_pos: ", eef_support_xy_pos)
        # centroids = eef_support_xy_pos.mean(dim=1) # NOTE: IF POSSIBLE MAYBE WILL NEED CONVEX HULL FOR EXTREME CASES(?)
        # print("centroids: ", centroids)
        # body_xys = params[:,18:20] # size = (batch_size, 2)
        # print("body_xys: ", body_xys)
        # body_xy_residual_squared = (body_xys - centroids) ** 2        

        # --- check pose validity ---
        pose_penalty = torch.tensor([0 if check_pose_validity(leg_pos=all_eef_pos_w[i,:,:3,3], 
                                                                body_pos=params[i,:3], legs_on_ground=legs_on_ground) 
                                        else 1e6 for i in range(batch_size)])
        print("pose_penalty: ", pose_penalty)

        

        # --- final cost function ---
        print("eef_pos_residual_squared: ", eef_pos_residual_squared.sum())
        # print("body_xy_residual_squared: ", body_xy_residual_squared.sum())
        print("free_legs_residual_squared: ", free_legs_residual_squared)

        cost = eef_pos_residual_squared.sum() + free_legs_residual_squared.sum() # + pose_penalty.sum()

        return cost

    res = minimize(
        cost_function, 
        params, 
        method='l-bfgs', 
        options=dict(line_search='strong-wolfe'),
        max_iter=50,
        disp=2
        )
    # print("result:\n", res.x)
    print("res.success: ", res.success)
    return res.x



      
## for visualization
cmap = colormaps['tab10']
colors = [cmap(i)[:3] for i in range(NUM_LEGS)]
colors_name = ["blue", "orange", "green", "red", "purple", "brown"]
mesh_name_left = ["M6_base_motor","M6_left_link1_red","M6_link2_red","M6_link3_red"]
mesh_name_right = ["M6_base_motor","M6_right_link1_red","M6_link2_red","M6_link3_red"]
mesh_base = os.path.join(curr_dir, "urdf", "yuna_stl", "M6_base_matt6_boxed_full.STL")
meshes_left = [trimesh.load_mesh(os.path.join(curr_dir, "urdf", "yuna_stl", f"{name}.STL")) for name in mesh_name_left]
meshes_right = [trimesh.load_mesh(os.path.join(curr_dir, "urdf", "yuna_stl", f"{name}.STL")) for name in mesh_name_right]

def visualize(base_trans, leg_trans, batch_size=batch_size, goal=None):
    scene = trimesh.Scene()
    spacing = 1.5  # spacing between each solution
    transformed_bases = []
    transformed_legs = []
    axes = []
    
    # Create ground plane
    ground_width, ground_height = 5.0, 5.0
    vertices = np.array([[-ground_width / 2, -ground_height / 2, 0], [ground_width / 2, -ground_height / 2, 0], 
                         [ground_width / 2, ground_height / 2, 0], [-ground_width / 2, ground_height / 2, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    ground_plane = trimesh.Trimesh(vertices=vertices, faces=faces)
    ground_plane.visual.face_colors = [0.5, 0.7, 1.0, 0.5]

    for k in range(batch_size):
        trans_dist = k * spacing
        translation = np.array([trans_dist, 0, 0])  # translate to place robot side by side
        
        # Add world origin axes
        axis_length = .5
        x_axis = trimesh.load_path([[trans_dist, 0, 0], [trans_dist + axis_length, 0, 0]])
        y_axis = trimesh.load_path([[trans_dist, 0, 0], [trans_dist, axis_length, 0]])
        z_axis = trimesh.load_path([[trans_dist, 0, 0], [trans_dist, 0, axis_length]])
        x_axis.colors  = [[255, 0, 0, 255] * len(x_axis.entities)]
        y_axis.colors  = [[0, 255, 0, 255] * len(x_axis.entities)]
        z_axis.colors  = [[0, 0, 255, 255] * len(x_axis.entities)]
        axes.extend([x_axis, y_axis, z_axis])

        transformed_base = mesh_base.copy()
        transformed_base.apply_transform(base_trans[k].detach().numpy())
        transformed_base.apply_translation(translation)
        transformed_bases.append(transformed_base)

        for i in range(NUM_LEGS):
            leg_meshes = meshes_left if i % 2 == 0 else meshes_right
            leg_transforms = leg_trans[k, i]
            color = colors[i]

            for j, mesh in enumerate(leg_meshes):
                transformed_mesh = mesh.copy()
                transformed_mesh.apply_transform(leg_transforms[j].detach().numpy())
                transformed_mesh.apply_translation(translation)
                transformed_mesh.visual.face_colors = [color] * len(transformed_mesh.faces)
                transformed_legs.append(transformed_mesh)

        if goal is not None:
            points = goal.numpy().reshape(-1, 3)
            spheres = [trimesh.creation.icosphere(radius=0.03).apply_translation(translation + point) for point in points]
            for sphere in spheres:
                sphere.visual.face_colors = [255, 0, 0, 200]
            scene.add_geometry(spheres)
        
    scene.add_geometry(ground_plane)
    scene.add_geometry(transformed_bases)
    scene.add_geometry(transformed_legs)
    scene.add_geometry(axes)
    scene.show()
    
    
if __name__=='__main__':
    
    ## --- Visualize initial base and legs pose ---
    # base_trans0, leg_trans = get_transformation_fk(th_0[0].reshape(1,18), batch_size=1)
    # print("base_trans0:\n", base_trans0.detach().numpy())
    # visualize(base_trans=base_trans0, leg_trans=leg_trans, batch_size=1, goal=None) # base origin = world origin
    
    # --- main ---
    # leg_idxs = [0, 1]
    # legs_on_ground = [False, False, True, True, True, True]
    # pos = torch.tensor([[0.51589,  0.23145, 0.2],
    #                     [0.51589, -0.23145, 0.2]])
    leg_idxs = [0, 1]
    legs_on_ground = [False, False, True, True, True, True]
    legs_plane = [PLANE2, PLANE2, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE, GROUND_PLANE]
    pos = torch.tensor([[0.51589, 0.23145, PLANE2],
                        [0.51589, -0.23145, PLANE2]])
    # leg_idxs = [0, 1, 2, 3, 4, 5]
    # pos = torch.tensor([[0.5, 0.3, 0], 
    #                     [0.5, -0.3, 0], 
    #                     [0.05, .5, 0],
    #                     [0.05, -.5, 0],
    #                     [-.45, 0.3, 0],
    #                     [-.45, -0.3, 0]])
    rot = torch.zeros_like(pos)
    goal = pk.Transform3d(pos=pos, rot=rot)
    params = solve_multiple_legs_ik(goal, legs_on_ground=legs_on_ground, legs_plane=legs_plane, leg_idxs=leg_idxs, batch_size=batch_size)
              
    robot_frame_trans, base_trans, leg_trans = get_transformations_from_params(params)
    
    # --- print solutions + visualization ---
    for i in range(batch_size):
        print(f"Solution {i + 1}:")
        print("robot_frame_trans:\n", robot_frame_trans[i].detach().numpy())
        print("base_trans:\n", base_trans[i].detach().numpy())
        for j in range(NUM_LEGS):
            print(f"leg{j}_trans:\n", leg_trans[i][j][-1].detach().numpy())
    
    visualize(base_trans=base_trans, leg_trans=leg_trans, goal=pos)

