import math
import pytorch_kinematics as pk
import torch
import trimesh
from matplotlib  import colormaps
from torchmin import minimize
import time
import numpy as np

NUM_Legs = 6
PI = math.pi
chains = [] # list to store chain for each leg
batch_size = 2
# initial joint angles for robot's legs (all legs on ground, joints at 90 deg)
# 1x18, each group of 3 values represents joint angles for each leg
th_0 = torch.tensor([0, 0, -PI / 2, 
                     0, 0, PI / 2, 
                     0, 0, -PI / 2,
                     0, 0, PI / 2, 
                     0, 0, -PI / 2, 
                     0, 0, PI / 2]*batch_size).reshape((batch_size, 18)) # TODO: remove batchsize from here

    
def get_transformation(theta):
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
    rob_tf = pk.Transform3d(pos=torch.cat([params[:,:2],torch.zeros(batch_size,1)],dim=-1), 
                            rot=torch.cat([torch.zeros(batch_size,2),params[:,2].unsqueeze(-1)],dim=-1))
    robot_frame_trans = rob_tf.get_matrix()
    base_trans, leg_trans = get_transformation(params[:,3:]) # in robot frame

    # transform to world frame
    base_trans = torch.bmm(robot_frame_trans,base_trans)
    leg_trans = torch.einsum('bkl,bijlm->bijkm',robot_frame_trans,leg_trans)    # einsum = Einstein summation notation, 
                                                                                # for specifying complex tensor operations with concise notation
    
    return robot_frame_trans, base_trans, leg_trans


def solve_single_leg_ik(goal, leg_idx, batch_size=1):
    # generate random initial guess for optimization
    xy = torch.rand(batch_size, 2) # xy base position
    z_rot = torch.rand(batch_size, 1) # z rotation angle of base
    theta = torch.rand(batch_size, 18) # joint angles
    params = torch.cat([xy,z_rot,theta], dim=1)
    
    def cost_function(params):
        # base_tf = Transform3d(pos=tensor([[x, y, 0]]), rot=tensor([[quaternion (w,x,y,z) of euler angle (0, 0, z_rot)]])) 
        #   torch.cat(..., dim=-1) concatenates input tensors along last dim
        #   .unsqueeze(dim) adds a new dim of size 1 at the specified position (for proper concatenation)
        rob_tf = pk.Transform3d(pos=torch.cat([params[:,:2],torch.zeros(batch_size,1)],dim=-1), 
                                    rot=torch.cat([torch.zeros(batch_size,2),params[:,2].unsqueeze(-1)],dim=-1))
        
        # get transformation of target leg's end-effector
        _,leg_trans = get_transformation(params[:,3:])
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
    print("result", res.x)
    return res.x


# for visualization (load meshes once)
cmap = colormaps['tab10']
colors = [cmap(i)[:3] for i in range(NUM_Legs)]
colors_name = ["blue", "orange", "green", "red", "purple", "brown"]
mesh_name_left = ["M6_base_motor","M6_left_link1_red","M6_link2_red","M6_link3_red"]
mesh_name_right = ["M6_base_motor","M6_right_link1_red","M6_link2_red","M6_link3_red"]
mesh_base = trimesh.load_mesh("urdf/yuna_stl/M6_base_matt6_boxed_full.STL")
meshes_left = [trimesh.load_mesh(f"urdf/yuna_stl/{name}.STL") for name in mesh_name_left]
meshes_right = [trimesh.load_mesh(f"urdf/yuna_stl/{name}.STL") for name in mesh_name_right]

def visualize(base_trans, leg_trans, goal=None):
    scene = trimesh.Scene()
    spacing = 1.5  # spacing between robots

    # Pre-allocate transformed meshes to reduce memory overhead
    transformed_bases = []
    transformed_legs = []

    for k in range(batch_size):  # batch size
        translation = np.array([k * spacing, 0, 0])  # translate to place robot side by side
        
        # Transform base mesh
        transformed_base = mesh_base.copy()
        transformed_base.apply_transform(base_trans[k].detach().numpy())
        transformed_base.apply_translation(translation)
        transformed_bases.append(transformed_base)

        for i in range(NUM_Legs):
            leg_meshes = meshes_left if i % 2 == 0 else meshes_right
            leg_transforms = leg_trans[k, i]
            color = colors[i]

            for j, mesh in enumerate(leg_meshes):
                transformed_mesh = mesh.copy()
                transformed_mesh.apply_transform(leg_transforms[j].detach().numpy())
                transformed_mesh.apply_translation(translation)
                transformed_mesh.visual.face_colors = [color] * len(transformed_mesh.faces)  # Use pre-defined color
                transformed_legs.append(transformed_mesh)

        # Add goal points if provided
        if goal is not None:
            points = goal.numpy().reshape(-1, 3)
            spheres = [trimesh.creation.icosphere(radius=0.03).apply_translation(translation + point) for point in points]
            for sphere in spheres:
                sphere.visual.face_colors = [255, 0, 0, 200]
            scene.add_geometry(spheres)
        
    # Add all transformed meshes to the scene at once
    scene.add_geometry(transformed_bases)
    scene.add_geometry(transformed_legs)

    scene.show()
    
    
if __name__=='__main__':
    base_trans0, leg_trans = get_transformation(th_0)

    # GOAL: minimize distance of leg 3 to goal
    leg_idx = 0
    pos = torch.tensor([0.5, 0.3, 0.0])
    rot = torch.tensor([0.0, 0.0, 0.0])
    goal = pk.Transform3d(pos=pos, rot=rot)
    print("goal", goal)
    
    params = solve_single_leg_ik(goal=goal, leg_idx=leg_idx, batch_size=batch_size)
    
    for i in range(NUM_Legs): # force all legs to be on ground (TODO: not the best way to do this)
        if i != leg_idx:
            params[:,3+3*i:3+3*(i+1)] = th_0[:,3*i:3*(i+1)]
      
    robot_frame_trans,base_trans,leg_trans = get_transformations_from_params(params)

    visualize(base_trans=base_trans, leg_trans=leg_trans, goal=pos)