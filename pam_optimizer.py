import os
import math
import pytorch_kinematics as pk
import torch
import trimesh
from matplotlib import colormaps
import matplotlib.pyplot as plt
from torchmin import minimize
import numpy as np
from static_stability_margin import convex_hull_pam, static_stability_margin, point_in_hull
import logging
from Yuna_Env import Map
import time

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

class PamOptimizer:
    def __init__(self, height_map=Map(), batch_size=1, vis=False):
        self.logger = self._init_logger()
        self.device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.chains = [] # list to store chain for each leg
        self.PI = math.pi
        self.NUM_LEGS = 6
        self.batch_size = batch_size
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.height_map = height_map
        self.BODY_HEIGHT_CLEARANCE_THRESHOLD = 0.12
        self.vis = vis
        if self.vis:
            self._load_vis_var()
        # initial guess (base at origin, eef at ~0.125m below base, last link perpendicular to ground)
        self.th_0 = torch.tensor([  0, -self.PI / 12, -self.PI * 7/12, 
                                    0,  self.PI / 12,  self.PI * 7/12, 
                                    0, -self.PI / 12, -self.PI * 7/12, 
                                    0,  self.PI / 12,  self.PI * 7/12, 
                                    0, -self.PI / 12, -self.PI * 7/12, 
                                    0,  self.PI / 12,  self.PI * 7/12] * batch_size).reshape((batch_size, 18)).to(self.device)
        

    def _init_logger(self):
        logger = logging.getLogger('pam_optimizer')
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('pam.log', mode='w')
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger

    def _load_vis_var(self):
            # load variables for visualization
            cmap = colormaps['tab10']
            self.colors = [cmap(i)[:3] for i in range(self.NUM_LEGS)]
            # self.colors_name = ["blue", "orange", "green", "red", "purple", "brown"]
            mesh_name_left = ["M6_base_motor","M6_left_link1_red","M6_link2_red","M6_link3_red"]
            mesh_name_right = ["M6_base_motor","M6_right_link1_red","M6_link2_red","M6_link3_red"]
            self.mesh_base = trimesh.load_mesh(os.path.join(self.curr_dir, "urdf", "yuna_stl", "M6_base_matt6_boxed_full.STL"))
            self.meshes_left = [trimesh.load_mesh(os.path.join(self.curr_dir, "urdf", "yuna_stl", f"{name}.STL")) for name in mesh_name_left]
            self.meshes_right = [trimesh.load_mesh(os.path.join(self.curr_dir, "urdf", "yuna_stl", f"{name}.STL")) for name in mesh_name_right]

    def get_transformation_fk(self, theta):
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
        theta = theta.to(dtype=torch.float64, device=self.device)
        
        for i in range(self.NUM_LEGS):
            # build serial chain for each leg, add to the list of chains
            chain = pk.build_serial_chain_from_urdf(open(os.path.join(self.curr_dir, "urdf", "yuna.urdf")).read().encode(),f"leg{i+1}__FOOT").to(dtype=torch.float64, device=self.device)
            # ^NOTE: suppressed "unknown tag warning" at /home/$USER/.local/lib/python3.10/site-packages/pytorch_kinematics/urdf_parser_py/xml_reflection/core.py
            self.chains.append(chain)
            
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
            leg_trans_r.append(trans_i.reshape(-1, self.batch_size, 4, 4).transpose(0, 1))
            # print(f"base_{i}: \n", base_trans_r)
            # print(f"trans_{i}: \n", trans_i)
        
        # reshape to dim [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4]
        leg_trans_r = torch.stack(leg_trans_r, dim=1).to(self.device)
        return base_trans_r, leg_trans_r # in self.device


    def get_transformations_from_params(self, params):
        """
        Computes base and leg transformation matrices from the given parameters.

        :param torch.Tensor params: A tensor containing joint angles and base position.
                                    Shape: [batch_size, 21], where the first 18 values are joint angles, 
                                    and the next 3 are base parameters (x,y,z) in world frame.
        :return: robot_frame_trans_w, base_trans_w, leg_trans_w, leg_trans_r
        - robot_frame_trans_w (torch.Tensor): Transformation matrix of robot frame in world frame. Shape: [batch_size, 4, 4].
        - base_trans_w (torch.Tensor): Base transformation matrix in world frame. Shape: [batch_size, 4, 4].
        - leg_trans_w (torch.Tensor): Leg transformation matrices in world frame. Shape: [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4].
        - leg_trans_r (torch.Tensor): Leg transformation matrices in robot frame. Shape: [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4].
        """
        params = params.to(self.device)
        # rob_tf = Transform3d(pos=tensor([[x, y, z]]), rot=tensor([[quaternion (w,x,y,z) of euler angle (0, 0, z_rot)]])) 
        # torch.cat(..., dim=-1) concatenates input tensors along last dim
        # .unsqueeze(dim) adds a new dim of size 1 at the specified position (for proper concatenation)
        rob_tf_w = pk.Transform3d(
            pos=torch.cat([params[:, 18:21]], dim=-1),  # extract base position (x, y, z)
            rot = torch.zeros(self.batch_size, 3, device=self.device), # no rotation at all (roll, pitch, yaw)
            # rot=torch.cat([torch.zeros(self.batch_size, 1, device=self.device),  # no x rotation (roll)
            #                 params[:,21].unsqueeze(1), # extract y rotation (pitch) of base
            #                 torch.zeros(self.batch_size, 1, device=self.device)], dim=-1), # no z rotation (yaw)
            device=self.device
        )        
        robot_frame_trans_w = rob_tf_w.get_matrix().to(torch.float64) # in world frame
        base_trans_r, leg_trans_r = self.get_transformation_fk(params[:,:18]) # in robot frame

        # transform to world frame
        base_trans_w = torch.bmm(robot_frame_trans_w,base_trans_r)
        leg_trans_w = torch.einsum('bkl,bijlm->bijkm', robot_frame_trans_w, leg_trans_r) # einsum = Einstein summation notation, for specifying complex tensor operations with concise notation
        
        return robot_frame_trans_w, base_trans_w, leg_trans_w, leg_trans_r


    def solve_multiple_legs_ik(self, pos, rot, leg_idxs, params=None, has_base_goal=False, target_base_xy=np.zeros(2), plot=False, plot_filename="pam_costs.png"):
        """
        Solves the inverse kinematics for multiple legs, optimizing to reach specified goal positions.

        :param pk.Transform3d goal: The desired position and orientation for the leg end-effectors. Shape: [len(leg_idxs), 4, 4].
        :param list leg_idxs: List of leg indices to be optimized.
        :param int batch_size: The number of samples processed at a time during a pass.

        :return: optimized_params
        - optimized_params (torch.Tensor): A tensor containing optimized joint angles and base position.
                                    Shape: [batch_size, 21], where the first 18 values are joint angles, 
                                    and the remaining 3 are base parameters (x,y,z) in world frame. 
        """        
        
        # generate random initial guess for optimization
        theta = self.th_0 # torch.rand(batch_size, 18) # joint angles
        base_xyz_w = torch.tensor([target_base_xy.tolist() + [self.BODY_HEIGHT_CLEARANCE_THRESHOLD]] * self.batch_size, device=self.device)
        # base_xyz_w = torch.rand(self.batch_size, 3, device=self.device) # xyz base position
        params = torch.cat([theta, base_xyz_w], dim=-1)

        # Initialize lists to track cost terms
        self.cost_history = []
        self.free_legs_history = []
        self.ssm_history = []
        self.body_height_history = []
        self.target_leg_history = []
        self.base_xy_history = []
        self.last_link_history = []

        def cost_function(params):
            rob_tf_w = pk.Transform3d(
                pos=torch.cat([params[:, 18:21]], dim=-1),  # extract base position (x, y, z)
                rot = torch.zeros(self.batch_size, 3, device=self.device), # no rotation at all (roll, pitch, yaw)
                device=self.device
            )
            # self.logger.debug(f"rob_tf_w: {rob_tf_w.get_matrix()}")
            base_trans_r, leg_trans_r = self.get_transformation_fk(params[:,:18]) # [batch_size, NUM_Legs, num_of_links_per_leg, 4, 4]
            eef_trans_r = leg_trans_r[:,:,-1,:,:] # [batch_size, NUM_Legs, 4, 4] (select last elem along 3rd dim)
            all_eef_pos_w = torch.einsum("bkl,bilm->bikm", rob_tf_w.get_matrix().to(torch.float64), eef_trans_r)
            target_eef_pos_w = all_eef_pos_w[:,leg_idxs,:3,3] # extract pos (xyz) of end-effectors
            
            # --- optimize based on static stability margin (min dist from the CoM to the edges of support polygon) ---
            self.logger.debug("\n--- optimizing based on static stability margin ---")
            # project all points to a ground plane and draw support polygon using legs on ground
            # then minimize distance of body CoM to the centroid of the support polygon
            eef_support_idxs = [i for i in range(self.NUM_LEGS) if i not in leg_idxs]
            eef_support_xy_pos = all_eef_pos_w[:, eef_support_idxs, :2, 3]
            body_xys = params[:,18:20] # size = (batch_size, 2)
            self.logger.debug(f"eef_support_idxs: {eef_support_idxs}")
            self.logger.debug(f"eef_support_xy_pos: {eef_support_xy_pos}")
            self.logger.debug(f"body_xys: {body_xys}")
            hull_points = convex_hull_pam(eef_support_xy_pos)
            ssm_cost = 0.1 * static_stability_margin(eef_support_xy_pos, body_xys).mean() # mean of all min distances from CoM to edges of support polygon
            self.logger.debug(f"ssm: {ssm_cost}")
            for i in range(body_xys.shape[0]): # Check if each body_xys point is inside the support polygon
                body_point = body_xys[i]
                if not point_in_hull(body_point, hull_points[i]):
                    ssm_cost -= 1e2  # Apply penalty for points outside the polygon
                    self.logger.warning("body xy not within support polygon")
            self.ssm_history.append(ssm_cost.cpu().detach().numpy())

            # --- optimize based on free legs' height ---
            # free legs = legs not specified for target goal pos (xyz), but heights are fixed to a specific plane
            # free_legs = [i for i in range(self.NUM_LEGS) if i not in leg_idxs]
            free_legs_height = all_eef_pos_w[:, eef_support_idxs, 2, 3] # free legs is support legs
            # TODO: detect if it is near any edges / steps (aka sudden height changes)
            heights_at_xy_pos_on_map = self.height_map.get_heights_at(eef_support_xy_pos.cpu().detach().numpy().reshape(-1, 2))
            free_legs_plane = torch.tensor(heights_at_xy_pos_on_map.reshape(self.batch_size, -1), device=self.device)
            free_legs_var = torch.tensor(self.height_map.get_variances_at(eef_support_xy_pos.cpu().detach().numpy().reshape(-1, 2)), device=self.device)
            free_legs_height_residual_squared = (free_legs_height - free_legs_plane) ** 2 # + free_legs_var ** 0.5
            self.logger.debug("\n--- optimizing based on free legs' height ---")
            self.logger.debug(f"free_legs_xy_pos:\n{eef_support_xy_pos}")
            self.logger.debug(f"free_legs_height:\n{free_legs_height}")
            self.logger.debug(f"free_legs_plane:\n{free_legs_plane}")
            self.logger.debug(f"free_legs_var:\n{free_legs_var}")
            self.logger.debug(f"free_legs_height_residual_squared:\n{free_legs_height_residual_squared}")
            self.free_legs_history.append(free_legs_height_residual_squared.sum().item())

            # --- optimize such that last link of supported legs is as perpendicular to ground as possible ---
            # when last link perpendicular to ground, x-axis of eef frame (pointing down) parallel to z-axis of world frame (pointing up)
            eef_trans_x_axis_w = all_eef_pos_w[:, eef_support_idxs, :3, 0] # extract x-axis of eef frame
            eef_trans_x_axis_ideal = torch.tensor([0., 0., -1.], device=self.device).repeat(self.batch_size, len(eef_support_idxs), 1)
            last_link_perpendicular_residual = (eef_trans_x_axis_w - eef_trans_x_axis_ideal) ** 2
            self.logger.debug("\n--- optimizing based on last link perpendicularity ---")
            self.logger.debug(f"x-axis of eef frame:\n{eef_trans_x_axis_w}")
            self.logger.debug(f"last_link_perpendicular_residual:\n{last_link_perpendicular_residual}")
            self.last_link_history.append(last_link_perpendicular_residual.sum().item())
            
            # --- final cost function ---
            cost = (
                free_legs_height_residual_squared.sum() 
                - ssm_cost 
                + last_link_perpendicular_residual.sum()
            )

            # --- optimize based on body height such that there is enough clearance below it ---
            # estimated body size: 390mm in y direction, 600mm in x direction
            base_center = params[:, 18:20] # (batch_size, 2)
            max_heights_below_body_area = torch.tensor([
                self.height_map.get_max_height_below_base(center[0].item(), center[1].item())
                for center in base_center
            ], device=self.device)
            body_height_clearance =  params[:, 20] - max_heights_below_body_area
            body_height_clearance_threshold = torch.tensor(self.BODY_HEIGHT_CLEARANCE_THRESHOLD, device=self.device).repeat(self.batch_size)
            
            self.logger.debug("\n--- optimizing based on body height ---")
            self.logger.debug(f"base_center: {base_center}")
            self.logger.debug(f"current_base_heights_guess: {params[:, 20]}")
            self.logger.debug(f"max_heights_below_body_area: {max_heights_below_body_area}")
            self.logger.debug(f"body_height_clearance: {body_height_clearance}")
            
            if torch.any(body_height_clearance < body_height_clearance_threshold):
                body_height_clearance_residual = (body_height_clearance_threshold - body_height_clearance) ** 2
                cost += body_height_clearance_residual.sum()
                self.logger.warning(f"body height is too low")
                self.logger.debug(f"body_height_clearance_residual: \n{body_height_clearance_residual}")

                self.body_height_history.append(body_height_clearance_residual.sum().item())
            else:
                self.logger.debug(f"body height is fine")

                self.body_height_history.append(0)
            
            # --- optimize based on distance between eef pos and goal pos ---
            has_eef_goal = len(pos) != 0
            if has_eef_goal:
                goal = pk.Transform3d(pos=pos.to(self.device), rot=rot.to(self.device), device=self.device)
                eef_pos_residual_squared = (target_eef_pos_w - goal.get_matrix()[:,:3,3].unsqueeze(0))**2 # calculate squared diff between eef pos and goal pos
                cost += eef_pos_residual_squared.sum()
                self.logger.debug("\n--- optimizing based on distance between eef pos and goal pos ---")
                self.logger.debug(f"eef_pos:\n{target_eef_pos_w}")
                self.logger.debug(f"goal_pos:\n{goal.get_matrix()[:,:3,3]}")
                self.logger.debug(f"eef_pos_residual_squared:\n{eef_pos_residual_squared}")
                self.target_leg_history.append(eef_pos_residual_squared.sum().item())

            if has_base_goal:
                target_base_xy_tensor = torch.tensor(target_base_xy, device=self.device).unsqueeze(0).repeat(self.batch_size, 1)
                base_xy_residual = (params[:, 18:20] - target_base_xy_tensor) ** 2
                cost += base_xy_residual.sum()
                self.logger.debug("\n--- optimizing based on base xy residual ---")
                self.logger.debug(f"target_base_xy: {target_base_xy}")
                self.logger.debug(f"current_base_xy_guess: {params[:, 18:20]}")
                self.logger.debug(f"base_xy_residual:\n{base_xy_residual}")
                self.base_xy_history.append(base_xy_residual.sum().item())

            self.cost_history.append(cost.item())
            return cost

        res = minimize(
            cost_function, 
            params, 
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=60,
            disp=2
            )
    
        self.logger.debug(f"success?: {res.success}")
        self.logger.debug(f"num of iterations: {res.nit}")
        self.logger.debug(f"functions eval: {res.nfev}")
        if res.success == False:
            self.logger.warning("Optimization did not converge")
        self.logger.debug(f"result:\n{res.x}")

        # self.plot_objective_history(plot_filename)
        self.plot_objective_history_one_graph(plot_filename)
        
        return res.x

    def plot_objective_history_one_graph(self, plot_filename="pam_costs.png"):
        start_plot_time = 5
        x_label = 'Time Step'
        y_label = 'Cost Value'
        
        plt.figure(figsize=(12, 10))
        
        plt.plot(self.cost_history[start_plot_time:], label='Total Cost', linewidth=2)
        plt.plot(self.free_legs_history[start_plot_time:], label='Free Legs Height', linestyle='--')
        plt.plot(self.ssm_history[start_plot_time:], label='Static Stability Margin', linestyle='--')
        plt.plot(self.last_link_history[start_plot_time:], label='Last Link Perpendicularity', linestyle='--')
        plt.plot(self.body_height_history[start_plot_time:], label='Body Height Clearance', linestyle='--')

        if len(self.target_leg_history) > 0:
            plt.plot(self.target_leg_history[start_plot_time:], label='Target Legs Position', linestyle='--')

        if len(self.base_xy_history) > 0:
            plt.plot(self.base_xy_history[start_plot_time:], label='Base XY', linestyle='--')

        plt.title(f'Convergence Plot ({plot_filename.split(".")[0]})')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.curr_dir, plot_filename))
        plt.close()
        # plt.show()

    def plot_objective_history(self, plot_filename="pam_costs.png"):
        start_plot_time = 5
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        x_label = 'Time Step'
        y_label = 'Cost Value'
        
        axs[0, 0].plot(self.cost_history[start_plot_time:], label='Total Cost', linewidth=2)
        axs[0, 0].set_title('Total Cost')
        axs[0, 0].set_xlabel(x_label)
        axs[0, 0].set_ylabel(y_label)
        # axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(self.free_legs_history[start_plot_time:], label='Free Legs Height', linestyle='--')
        axs[0, 1].set_title('Free Legs Height')
        axs[0, 1].set_xlabel(x_label)
        axs[0, 1].set_ylabel(y_label)
        # axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        axs[1, 0].plot(self.ssm_history[start_plot_time:], label='Static Stability Margin', linestyle='--')
        axs[1, 0].set_title('Static Stability Margin')
        axs[1, 0].set_xlabel(x_label)
        axs[1, 0].set_ylabel(y_label)
        # axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].plot(self.last_link_history[start_plot_time:], label='Last Link Perpendicularity', linestyle='--')
        axs[1, 1].set_title('Last Link Perpendicularity')
        axs[1, 1].set_xlabel(x_label)
        axs[1, 1].set_ylabel(y_label)
        # axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        axs[2, 0].plot(self.body_height_history[start_plot_time:], label='Body Height Clearance', linestyle='--')
        axs[2, 0].set_title('Body Height Clearance')
        axs[2, 0].set_xlabel(x_label)
        axs[2, 0].set_ylabel(y_label)
        # axs[2, 0].legend()
        axs[2, 0].grid(True)
        
        if len(self.target_leg_history) > 0:
            axs[2, 1].plot(self.target_leg_history[start_plot_time:], label='Target Legs Position', linestyle='--')
            axs[2, 1].set_title('Target Legs Position')
            axs[2, 1].set_xlabel(x_label)
            axs[2, 1].set_ylabel(y_label)
            # axs[2, 1].legend()
            axs[2, 1].grid(True)
        
        if len(self.base_xy_history) > 0:
            axs[3, 0].plot(self.base_xy_history[start_plot_time:], label='Base XY', linestyle='--')
            axs[3, 0].set_title('Base XY')
            axs[3, 0].set_xlabel(x_label)
            axs[3, 0].set_ylabel(y_label)
            # axs[3, 0].legend()
            axs[3, 0].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.curr_dir, plot_filename))
        plt.close()
        # plt.show()


    def visualize(self, base_trans, leg_trans, goal=None):        
        # make sure base_trans and leg_trans in numpy
        if self.vis == False:
            self.vis = True
            self._load_vis_var()

        scene = trimesh.Scene()
        spacing = 1.5  # spacing between each solution
        transformed_bases = []
        transformed_legs = []
        axes = []
        
        # Create ground plane
        # ground_width, ground_height = 5.0, 5.0
        # vertices = np.array([[-ground_width / 2, -ground_height / 2, 0], [ground_width / 2, -ground_height / 2, 0], 
        #                     [ground_width / 2, ground_height / 2, 0], [-ground_width / 2, ground_height / 2, 0]])
        # faces = np.array([[0, 1, 2], [0, 2, 3]])
        # ground_plane = trimesh.Trimesh(vertices=vertices, faces=faces)
        # ground_plane.visual.face_colors = [0.5, 0.7, 1.0, 0.5]

        # translate to place robot side by side
        for k in range(self.batch_size):
            trans_dist = k * spacing
            translation = np.array([trans_dist, 0, 0]) # translation for each solution       
            
            # Add world origin axes
            axis_length = .5
            x_axis = trimesh.load_path([[trans_dist, 0, 0], [trans_dist + axis_length, 0, 0]])
            y_axis = trimesh.load_path([[trans_dist, 0, 0], [trans_dist, axis_length, 0]])
            z_axis = trimesh.load_path([[trans_dist, 0, 0], [trans_dist, 0, axis_length]])
            x_axis.colors  = [[255, 0, 0, 255] * len(x_axis.entities)]
            y_axis.colors  = [[0, 255, 0, 255] * len(x_axis.entities)]
            z_axis.colors  = [[0, 0, 255, 255] * len(x_axis.entities)]
            axes.extend([x_axis, y_axis, z_axis])

            transformed_base = self.mesh_base.copy()
            transformed_base.apply_transform(base_trans[k])
            transformed_base.apply_translation(translation)
            transformed_bases.append(transformed_base)

            for i in range(self.NUM_LEGS):
                leg_meshes = self.meshes_left if i % 2 == 0 else self.meshes_right
                leg_transforms = leg_trans[k, i]
                color = self.colors[i]

                for j, mesh in enumerate(leg_meshes):
                    transformed_mesh = mesh.copy()
                    transformed_mesh.apply_transform(leg_transforms[j])
                    transformed_mesh.apply_translation(translation)
                    transformed_mesh.visual.face_colors = [color] * len(transformed_mesh.faces)
                    transformed_legs.append(transformed_mesh)

            if goal is not None:
                points = goal.numpy().reshape(-1, 3)
                spheres = [trimesh.creation.icosphere(radius=0.03).apply_translation(translation + point) for point in points]
                for sphere in spheres:
                    sphere.visual.face_colors = [255, 0, 0, 200]
                scene.add_geometry(spheres)

        # scene.add_geometry(ground_plane)
        scene.add_geometry(transformed_bases)
        scene.add_geometry(transformed_legs)
        scene.add_geometry(axes)
        scene.show()

  

if __name__=='__main__':

    height_map = Map()
    # height_map.load_map("fyp_height_map")
    # height_map.plot()

    optimizer = PamOptimizer(height_map=height_map, vis=True, batch_size=1)

    ## --- Visualize initial base and legs pose ---
    base_trans0, leg_trans0 = optimizer.get_transformation_fk(optimizer.th_0)
    # eef_trans0 = leg_trans0[:,:,-1,:,:]
    # print("base_trans0:\n", base_trans0.detach().numpy())
    # print("leg_trans0:\n", leg_trans0)
    # print("eef_trans:\n", eef_trans0)
    # print("eef_pos:\n", eef_trans0[:, :, :3, 3])
    # base_trans0 = base_trans0.cpu().detach().numpy()
    # leg_trans0 = leg_trans0.cpu().detach().numpy()
    # optimizer.visualize(base_trans=base_trans0, leg_trans=leg_trans0, goal=None) # base origin = world origin

    ## --- test get_transformations_from_params --- 
    # theta = optimizer.th_0.to(optimizer.device) # joint angles
    # params = torch.cat([theta, torch.zeros(optimizer.batch_size, 4).to(optimizer.device)], dim=-1)
    # transform = optimizer.get_transformations_from_params(params)
    # print(f"transform: {transform}")

    # leg_idxs = [1]
    # pos = torch.tensor([[1.4, -0.3, 0.7]])

    leg_idxs = [0, 1]
    pos = torch.tensor([[1.4, 0.3, 0.7], [1.4, -0.3, 0.6]])

    rot = torch.zeros_like(pos)

    start_time = time.time()
    params = optimizer.solve_multiple_legs_ik(pos, rot, leg_idxs=leg_idxs)
    robot_frame_trans_w, base_trans_w, leg_trans_w, leg_trans_r = optimizer.get_transformations_from_params(params)
    print("Time taken: ", time.time() - start_time)
    
    robot_frame_trans_w = robot_frame_trans_w.cpu().detach().numpy()
    base_trans_w = base_trans_w.cpu().detach().numpy()
    leg_trans_w = leg_trans_w.cpu().detach().numpy()
    # # --- print solutions + visualization ---
    for i in range(optimizer.batch_size):
        print(f"Solution {i + 1}:")
        print("robot_frame_trans:\n", robot_frame_trans_w[i])
        print("base_trans:\n", base_trans_w[i])
        for j in range(optimizer.NUM_LEGS):
            print(f"leg{j}_trans:\n", leg_trans_w[i][j][-1])
    
    optimizer.visualize(base_trans=base_trans_w, leg_trans=leg_trans_w, goal=pos)

