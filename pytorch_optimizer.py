import math
import pytorch_kinematics as pk
import torch
import trimesh
from matplotlib  import colormaps
from torchmin import minimize
import sys
import numpy as np

NUM_Legs = 6
PI = math.pi
chains = [] # list to store chain for each leg
batch_size = 2
np.set_printoptions(precision=3, suppress=True)

# initial [guess] joint angles for robot's legs (all legs on ground, joints at 90 deg)
# 1x18, each group of 3 values represents joint angles for each leg
th_0 = torch.tensor([0, 0, -PI / 2, 
                     0, 0, PI / 2, 
                     0, 0, -PI / 2,
                     0, 0, PI / 2, 
                     0, 0, -PI / 2, 
                     0, 0, PI / 2]*batch_size).reshape((batch_size, 18))

    
def get_transformation(theta, batch_size=batch_size):
    """
    Computes the transformation matrices for each leg based on the given joint angles.
    
    :params torch.Tensor theta: The joint angles of all legs. Shape: [batch_size, 18].
    :params int, optional batch_size: The number of samples processed at a time during a pass. 
    
    :return: base_trans, leg_trans
    - base_trans (torch.Tensor): Transformation matrix of the body/base. Shape: [batch_size, 4, 4].
    - leg_trans (torch.Tensor): Transformation matrices of each leg. Shape: [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4].
    """
    leg_trans = []
    base_trans = []
    
    for i in range(NUM_Legs):
        # build serial chain for each leg, add to the list of chains
        chain = pk.build_serial_chain_from_urdf(open("urdf/yuna.urdf").read().encode(),f"leg{i+1}__FOOT")
        chains.append(chain)
        
        # extract joint angles for each leg
        theta_i = theta[:, 3*i:3*(i+1)]
        
        # perform FK for each leg, ret contains trans mat of all links of that leg
        ret = chain.forward_kinematics(theta_i, end_only=False)
        
        # extract trans mat for base link (should be same for all legs)
        if i == 0:
            base_trans = ret['base_link'].get_matrix()
        
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
        leg_trans.append(trans_i.reshape(-1, batch_size, 4, 4).transpose(0, 1))
        # print(f"base_{i}: \n", base_trans)
        # print(f"trans_{i}: \n", trans_i)
    
    # reshape to dim [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4]
    leg_trans = torch.cat(leg_trans).reshape(NUM_Legs, batch_size, -1, 4, 4).transpose(0, 1)
    return base_trans, leg_trans


def get_transformations_from_params(params):
    """
    Computes the base and leg transformation matrices from the given parameters.

    :param torch.Tensor params: A tensor containing the base position, rotation, and joint angles.
                                Shape: [batch_size, 22], where the first 3 values are base parameters (x, y, z, z_rot) 
                                and the remaining 18 are joint angles. 

    :return: robot_frame_trans, base_trans, leg_trans
    - robot_frame_trans (torch.Tensor): Transformation matrix of the robot frame in the world frame. Shape: [batch_size, 4, 4].
    - base_trans (torch.Tensor): Base transformation matrix in the world frame. Shape: [batch_size, 4, 4].
    - leg_trans (torch.Tensor): Leg transformation matrices in the world frame. Shape: [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4].
    """
    rob_tf = pk.Transform3d(pos=torch.cat([params[:,:3]],dim=-1), 
                            rot=torch.cat([torch.zeros(batch_size,2),params[:,3].unsqueeze(-1)],dim=-1))
    robot_frame_trans = rob_tf.get_matrix()
    base_trans, leg_trans = get_transformation(params[:,4:]) # in robot frame

    # transform to world frame
    base_trans = torch.bmm(robot_frame_trans,base_trans)
    leg_trans = torch.einsum('bkl,bijlm->bijkm',robot_frame_trans,leg_trans) # einsum = Einstein summation notation, for specifying complex tensor operations with concise notation
    
    return robot_frame_trans, base_trans, leg_trans


def solve_single_leg_ik(goal, leg_idx, batch_size=1):
    """
    Solves the inverse kinematics for a single leg, optimizing to reach a specified goal position.

    :param pk.Transform3d goal: The desired position and orientation for the leg end-effector.
    :param int leg_idx: The index of the leg to be optimized.
    :param int batch_size: The number of samples processed at a time during a pass.

    :return: optimized_params
    - optimized_params (torch.Tensor): Shape: [batch_size, 21], where the first 3 values are base parameters (x, y, z_rot) 
            and the remaining 18 are joint angles. 
    """
    # generate random initial guess for optimization
    xyz = torch.rand(batch_size, 3) # xy base position
    z_rot = torch.rand(batch_size, 1) # z rotation angle of base
    theta = th_0 # torch.rand(batch_size, 18) # joint angles
    params = torch.cat([xyz,z_rot,theta], dim=-1)
    
    def cost_function(params):
        # base_tf = Transform3d(pos=tensor([[x, y, z]]), rot=tensor([[quaternion (w,x,y,z) of euler angle (0, 0, z_rot)]])) 
        #   torch.cat(..., dim=-1) concatenates input tensors along last dim
        #   .unsqueeze(dim) adds a new dim of size 1 at the specified position (for proper concatenation)
        rob_tf = pk.Transform3d(pos=torch.cat([params[:,:3]],dim=-1), 
                                    rot=torch.cat([torch.zeros(batch_size,2),params[:,3].unsqueeze(-1)],dim=-1))
        
        # get transformation of target leg's end-effector
        _,leg_trans = get_transformation(params[:,4:])
        leg_i_eef_trans = leg_trans[:,leg_idx,-1,:,:]
        
        # combine base_tf (guess) with leg's end-effector transformation
        leg_i_eef = torch.bmm(rob_tf.get_matrix(),leg_i_eef_trans) # bmm = batch matrix multiplication
        eef_pos = leg_i_eef[:,:3,3] # extract pos (xyz) of end-effector
        
        # calculate squared diff between eef pos and goal pos
        residual_squared = (eef_pos - goal.get_matrix()[:,:3,3])**2 # shape = [batch_size, 3]
        cost = residual_squared.sum()
        return cost

    res = minimize(
        cost_function, 
        params, 
        method='l-bfgs', 
        options=dict(line_search='strong-wolfe'),
        max_iter=50,
        disp=2
        )    
    print("result\n", res.x)
    print("res.success: ", res.success)
    return res.x

def solve_multiple_legs_ik(goal, leg_idxs, batch_size=1):
    """
    Solves the inverse kinematics for multiple legs, optimizing to reach specified goal positions.

    :param pk.Transform3d goal: The desired position and orientation for the leg end-effectors. Shape: [len(leg_idxs), 4, 4].
    :param list leg_idxs: List of leg indices to be optimized.
    :param int batch_size: The number of samples processed at a time during a pass.

    :return: optimized_params
    - optimized_params (torch.Tensor): Shape: [batch_size, 21], where the first 4 values are base parameters (x, y, z, z_rot) 
            and the remaining 18 are joint angles. 
    """
    assert goal.get_matrix().shape[0] == len(leg_idxs)
    
    # generate random initial guess for optimization
    xyz = torch.rand(batch_size, 3) # xyz base position
    z_rot = torch.rand(batch_size, 1) # z rotation angle of base
    theta = th_0 # torch.rand(batch_size, 18) # joint angles
    params = torch.cat([xyz,z_rot,theta], dim=-1)
    
    def cost_function(params):
        rob_tf = pk.Transform3d(pos=torch.cat([params[:,:3]],dim=-1), 
                                    rot=torch.cat([torch.zeros(batch_size,2),params[:,3].unsqueeze(-1)],dim=-1))

        _, leg_trans = get_transformation(params[:,4:])
        leg_i_eef_trans = leg_trans[:,leg_idxs,-1,:,:]
        leg_i_eef = torch.einsum("bkl,bilm->bikm",rob_tf.get_matrix(),leg_i_eef_trans)
        eef_pos = leg_i_eef[:,:,:3,3]
        
        # optimize based on distance between eef pos and goal pos
        residual_squared = (eef_pos - goal.get_matrix()[:,:3,3].unsqueeze(0))**2
        
        cost = residual_squared.sum()
        return cost
    
    res = minimize(
        cost_function, 
        params, 
        method='l-bfgs', 
        options=dict(line_search='strong-wolfe'),
        max_iter=50,
        disp=2
        )
    print("result:\n", res.x)
    print("res.success: ", res.success)
    return res.x


def solve_all_legs_ik(goal, batch_size=1):
    ''' :param goal: target position for all legs, shape = [NUM_Legs, 4, 4]
    '''
    assert goal.get_matrix().shape[0] == NUM_Legs
    
    # generate random initial guess for optimization
    xyz = torch.rand(batch_size, 3) # xyz base position
    z_rot = torch.rand(batch_size, 1) # z rotation angle of base
    theta = th_0 # torch.rand(batch_size, 18) # joint angles
    params = torch.cat([xyz,z_rot,theta], dim=-1)
    
    def cost_function(params):
        rob_tf = pk.Transform3d(pos=torch.cat([params[:,:3]],dim=-1), 
                                    rot=torch.cat([torch.zeros(batch_size,2),params[:,3].unsqueeze(-1)],dim=-1))

        _, leg_trans = get_transformation(params[:,4:])
        leg_i_eef_trans = leg_trans[:,list(range(6)),-1,:,:]
        leg_i_eef = torch.einsum("bkl,bilm->bikm",rob_tf.get_matrix(),leg_i_eef_trans)
        eef_pos = leg_i_eef[:,:,:3,3]
        
        # optimize based on distance between eef pos and goal pos
        residual_squared = (eef_pos - goal.get_matrix()[:,:3,3].unsqueeze(0))**2
        
        cost = residual_squared.sum()
        return cost
    
    res = minimize(
        cost_function, 
        params, 
        method='l-bfgs', 
        options=dict(line_search='strong-wolfe'),
        max_iter=50,
        disp=2
        )
    print("result:\n", res.x)
    print("res.success: ", res.success)
    return res.x

        
## for visualization (load meshes once)
cmap = colormaps['tab10']
colors = [cmap(i)[:3] for i in range(NUM_Legs)]
colors_name = ["blue", "orange", "green", "red", "purple", "brown"]
mesh_name_left = ["M6_base_motor","M6_left_link1_red","M6_link2_red","M6_link3_red"]
mesh_name_right = ["M6_base_motor","M6_right_link1_red","M6_link2_red","M6_link3_red"]
mesh_base = trimesh.load_mesh("urdf/yuna_stl/M6_base_matt6_boxed_full.STL")
meshes_left = [trimesh.load_mesh(f"urdf/yuna_stl/{name}.STL") for name in mesh_name_left]
meshes_right = [trimesh.load_mesh(f"urdf/yuna_stl/{name}.STL") for name in mesh_name_right]

def visualize(base_trans, leg_trans, batch_size=batch_size, goal=None):
    scene = trimesh.Scene()
    spacing = 1.5  # spacing between robots
    transformed_bases = []
    transformed_legs = []
    axes = []
    
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

        # Transform base mesh
        transformed_base = mesh_base.copy()
        transformed_base.apply_transform(base_trans[k].detach().numpy())
        transformed_base.apply_translation(translation)
        transformed_bases.append(transformed_base)

        for i in range(NUM_Legs):
            leg_meshes = meshes_left if i % 2 == 0 else meshes_right
            leg_transforms = leg_trans[k, i]
            color = colors[i]

            # Transform leg meshes
            for j, mesh in enumerate(leg_meshes):
                transformed_mesh = mesh.copy()
                transformed_mesh.apply_transform(leg_transforms[j].detach().numpy())
                transformed_mesh.apply_translation(translation)
                transformed_mesh.visual.face_colors = [color] * len(transformed_mesh.faces)
                transformed_legs.append(transformed_mesh)

        # Add goal points if provided
        if goal is not None:
            points = goal.numpy().reshape(-1, 3)
            spheres = [trimesh.creation.icosphere(radius=0.03).apply_translation(translation + point) for point in points]
            for sphere in spheres:
                sphere.visual.face_colors = [255, 0, 0, 200]
            scene.add_geometry(spheres)
        
    # Add all meshes to the scene at once
    scene.add_geometry(transformed_bases)
    scene.add_geometry(transformed_legs)
    scene.add_geometry(axes)
    
    scene.show()
    
    
if __name__=='__main__':
    
    ## Visualize initial base and legs pose
    # base_trans0, leg_trans = get_transformation(th_0[0].reshape(1,18), batch_size=1)
    # visualize(base_trans=base_trans0, leg_trans=leg_trans, batch_size=1, goal=None)
    
    valid_modes = ["all_legs", "single_leg", "multiple_legs"]
    solve_mode = "all_legs"
    if solve_mode not in valid_modes:
        print(f"Error: Invalid solve mode '{solve_mode}'. Valid options are: {', '.join(valid_modes)}.")
        sys.exit(1) 
    
    if solve_mode == "all_legs":
        pos = torch.tensor([[0.51589, 0.23145, 0.3],
                            [0.51589, -0.23145, 0],
                            [0.0575, 0.5125, 0],
                            [0.0575, -0.5125, 0.3],
                            [-0.45839, 0.33105, 0.3],
                            [-0.45839, -0.33105, 0]])
        rot = torch.zeros_like(pos)
        goal = pk.Transform3d(pos=pos, rot=rot)        
        params = solve_all_legs_ik(goal, batch_size=batch_size)    
    
    elif solve_mode == "single_leg":
        leg_idx = 0
        pos = torch.tensor([0.3, 0.3, -0.1])
        rot = torch.tensor([0.0, 0.0, 0.0])
        goal = pk.Transform3d(pos=pos, rot=rot)
        params = solve_single_leg_ik(goal=goal, leg_idx=leg_idx, batch_size=batch_size)

    elif solve_mode == "multiple_legs":
        leg_idxs = [0, 1]
        pos = torch.tensor([[0.5, 0.3, -0.1], 
                            [0.5, -0.3, 0]])
        # leg_idxs = [0, 1, 2, 3, 4, 5]
        # pos = torch.tensor([[0.5, 0.3, 0], 
        #                     [0.5, -0.3, 0], 
        #                     [0.05, .5, 0],
        #                     [0.05, -.5, 0],
        #                     [-.45, 0.3, 0],
        #                     [-.45, -0.3, 0]])
        rot = torch.zeros_like(pos)
        goal = pk.Transform3d(pos=pos, rot=rot)
        params = solve_multiple_legs_ik(goal, leg_idxs=leg_idxs, batch_size=batch_size)
    
    # for i in range(NUM_Legs): # force all remaining legs (not involved in optimization) to be at initial pos (for temporary debugging, may not the best way to do this)
    #     if (solve_single_leg and i != leg_idx) or (not solve_single_leg and i not in leg_idxs):
    #         params[:,3+3*i:3+3*(i+1)] = th_0[:,3*i:3*(i+1)]
      
    robot_frame_trans, base_trans, leg_trans = get_transformations_from_params(params)
    
    for i in range(batch_size):
        print(f"Solution {i + 1}:")
        print("robot_frame_trans:\n", robot_frame_trans[i].detach().numpy())
        print("base_trans:\n", base_trans[i].detach().numpy())
        for j in range(NUM_Legs):
            print(f"leg{j}_trans:\n", leg_trans[i][j][-1].detach().numpy())
    
    visualize(base_trans=base_trans, leg_trans=leg_trans, goal=pos)

