# optimizer code shared by sun ge
# got error ahhh

import math
import pytorch_kinematics as pk
import torch
import trimesh
from matplotlib.cm import get_cmap
from torchmin import minimize
import time

NUM_DoFs = 18
NUM_Legs = 6
PI = math.pi
chain = pk.build_serial_chain_from_urdf(open("urdf/yuna.urdf").read().encode(),"leg3__FOOT")
joint_lim = torch.tensor(chain.get_joint_limits())
th_0 = torch.tensor([0, 0, -PI / 2, 0, 0, -PI / 2, 0, 0, -PI / 2,
                       0, 0, PI / 2, 0, 0, PI / 2, 0, 0, PI / 2]).reshape((1, 18))

def get_transformation(theta):
    B = theta.shape[0]
    ret = chain.forward_kinematics(theta, end_only=False)
    # calculate the transformations
    base_trans = ret['base_link'].get_matrix()
    camera_trans = ret['camera_mount'].get_matrix()
    leg_trans = []
    for i in range(NUM_Legs):
        mat_base = ret[f'base{i+1}__INPUT_INTERFACE'].get_matrix()
        mat_shoulder = ret[f'shoulder{i+1}__INPUT_INTERFACE'].get_matrix()
        mat_elbow = ret[f'elbow{i+1}__INPUT_INTERFACE'].get_matrix()
        mat_leg = ret[f'leg{i+1}__LAST_LINK'].get_matrix()
        mat_foot = ret[f'leg{i+1}__FOOT'].get_matrix()
        trans_i = torch.cat([mat_base, mat_shoulder, mat_elbow, mat_leg, mat_foot])
        leg_trans.append(trans_i.reshape(-1,B,4,4).transpose(0,1))
    leg_trans = torch.cat(leg_trans).reshape(NUM_Legs,B,-1, 4, 4).transpose(0,1)
    return base_trans, camera_trans, leg_trans


def solve_single_leg_ik(goal,batch=10,leg_idx=0):

    theta = torch.rand(batch,18)
    xy = torch.rand(batch,2)
    z_rot = torch.rand(batch,1)
    params = torch.cat([xy,z_rot,theta],dim=-1)

    def cost_function(params):
        rob_tf = pk.Transform3d(pos=torch.cat([params[:,:2],torch.zeros(batch,1)],dim=-1), 
                                rot=torch.cat([torch.zeros(batch,2),params[:,2].unsqueeze(-1)],dim=-1))
        
        _,_,leg_trans = get_transformation(params[:,3:])
        leg_i_eef_trans = leg_trans[:,leg_idx,-1,:,:]
        
        leg_i_eef = torch.bmm(rob_tf.get_matrix(),leg_i_eef_trans)
        eef_pos = leg_i_eef[:,:3,3]
        residual_squared =(eef_pos - goal.get_matrix()[:,:3,3])**2
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
    return res.x

def solve_multiple_legs_ik(goal,batch=10,leg_idx=[0,1]):

    assert goal.get_matrix().shape[0] == len(leg_idx)

    theta = torch.rand(batch,18)
    xy = torch.rand(batch,2)
    z_rot = torch.rand(batch,1)
    params = torch.cat([xy,z_rot,theta],dim=-1)

    def cost_function(params):
        rob_tf = pk.Transform3d(pos=torch.cat([params[:,:2],torch.zeros(batch,1)],dim=-1), 
                                rot=torch.cat([torch.zeros(batch,2),params[:,2].unsqueeze(-1)],dim=-1))
        
        _,_,leg_trans = get_transformation(params[:,3:])
        leg_i_eef_trans = leg_trans[:,leg_idx,-1,:,:]
        leg_i_eef = torch.einsum("bkl,bilm->bikm",rob_tf.get_matrix(),leg_i_eef_trans)
        eef_pos = leg_i_eef[:,:,:3,3]

        residual_squared =(eef_pos - goal.get_matrix()[:,:3,3].unsqueeze(0))**2
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
    return res.x

def get_transformations_from_params(params):
    B = params.shape[0]
    rob_tf = pk.Transform3d(pos=torch.cat([params[:,:2],torch.zeros(B,1)],dim=-1), 
                            rot=torch.cat([torch.zeros(B,2),params[:,2].unsqueeze(-1)],dim=-1))
    robot_frame_trans = rob_tf.get_matrix()
    base_trans,camera_trans,leg_trans = get_transformation(params[:,3:])

    base_trans = torch.bmm(robot_frame_trans,base_trans)
    camera_trans = torch.bmm(robot_frame_trans,camera_trans)
    leg_trans = torch.einsum('bkl,bijlm->bijkm',robot_frame_trans,leg_trans)
    return robot_frame_trans,base_trans,camera_trans,leg_trans

solve_single_leg = False
t0 = time.time()
if solve_single_leg:
    pos = torch.tensor([0.5, 0.3, 0.0])
    rot = torch.tensor([0.0, 0.0, 0.0])
    goal = pk.Transform3d(pos=pos, rot=rot)
    params = solve_single_leg_ik(goal)
else:
    pos = torch.tensor([[0.5, 0.3, -0.2],[0.8, 0.1, -0.2]])
    rot = torch.zeros_like(pos)
    goal = pk.Transform3d(pos=pos, rot=rot)
    # define a ball based on goal for visualization 
    ball_center = torch.mean(pos,dim=0)
    ball_radius = torch.max(torch.norm(pos - ball_center,dim=-1))
    params = solve_multiple_legs_ik(goal,batch=100,leg_idx=[0,1])
    # fix other leg theta
    params[:,6:12] = th_0[:,3:9]
    params[:,15:] = th_0[:,12:]
print(f'find {len(params)} solutions in {time.time()-t0} seconds')

# check optimized solutions
robot_frame_trans,base_trans,camera_trans,leg_trans = get_transformations_from_params(params)
# print(robot_frame_trans.shape,
#       base_trans.shape,
#       camera_trans.shape,
#       leg_trans.shape)

# visualize optimized solutions
# create color map for 6 legs
cmap = get_cmap("tab10", NUM_Legs)
# load meshes for visualization
mesh_name_left = ["M6_base_motor","M6_left_link1_red","M6_link2_red","M6_link3_red"]
mesh_name_right = ["M6_base_motor","M6_left_link1_red","M6_link2_red","M6_link3_red"]

scene = trimesh.Scene()
# visulize each goal point as a sphere
for point in pos.numpy().reshape(-1,3):
    sphere = trimesh.creation.icosphere(radius=0.03).apply_translation(point)
    sphere.visual.face_colors = [255,0,0,200]
    scene.add_geometry(sphere)
ball = trimesh.creation.icosphere(radius=ball_radius.item()).apply_translation(ball_center.numpy())
ball.visual.face_colors = [100,100,100,100]
scene.add_geometry(ball)

# number k solution 
for k in range(len(params)):
    base = trimesh.load_mesh("urdf/yuna_stl/M6_base_matt6_boxed_full.STL")
    base.apply_transform(base_trans[k].detach().numpy())
    scene.add_geometry(base)
    camera = trimesh.load_mesh("urdf/yuna_stl/camera_mount.STL")
    camera.apply_transform(camera_trans[k].detach().numpy())
    scene.add_geometry(camera)

    for i in range(NUM_Legs):
        if i % 2 == 0:
            for j in range(4):
                mesh = trimesh.load_mesh(f"urdf/yuna_stl/{mesh_name_left[j]}.STL")
                mesh.apply_transform(leg_trans[k,i,j].detach().numpy())
                mesh.visual.face_colors = [cmap(i)[:3]] * len(mesh.faces)
                scene.add_geometry(mesh)
        if i % 2 == 1:
            for j in range(4):
                mesh = trimesh.load_mesh(f"urdf/yuna_stl/{mesh_name_right[j]}.STL")
                mesh.apply_transform(leg_trans[k,i,j].detach().numpy())
                mesh.visual.face_colors = [cmap(i)[:3]] * len(mesh.faces)
                scene.add_geometry(mesh)
    scene.show()
