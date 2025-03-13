from Yuna_TrajPlanner import TrajPlanner
from Yuna_Env import YunaEnv
import numpy as np
from functions import transxy, solveFK, rotx
import time
import torch
from pam_optimizer import PamOptimizer
import matplotlib.pyplot as plt

STEP_HEIGHT = 0.08
GROUND_PLANE = 0.0
NUM_LEGS = 6
h = 0.12
RAISE_THRES = 0.1

eePos = np.array(  [[0.51589,    0.51589,   0.0575,     0.0575,     -0.45839,   -0.45839],
                    [0.23145,   -0.23145,   0.5125,     -0.5125,    0.33105,    -0.33105],
                    [   -h,         -h,         -h,         -h,         -h,         -h]])
eePos0_opt = np.array( [[ 0.4901,  0.4900,  0.0426,  0.0424, -0.4475, -0.4476],
                        [ 0.2338, -0.2340,  0.4914, -0.4914,  0.3076, -0.3074],
                        [-0.1252, -0.1252, -0.1252, -0.1252, -0.1252, -0.1252] ])

class Yuna:
    def __init__(self, visualiser=True, camerafollow=True, real_robot_control=False, pybullet_on=True, show_ref_points=False, 
                    eePos=eePos0_opt, goal=[], load_fyp_map=False,
                    batch_size=1, opt_vis=False, body_interval=0.15):
        # initialise the environment
        self.env = YunaEnv(visualiser=visualiser, camerafollow=camerafollow, real_robot_control=real_robot_control, pybullet_on=pybullet_on, 
                                eePos=eePos, goal=goal, load_fyp_map=load_fyp_map)
        self.bodyPos = self.env.body_pos_w.copy()   # initial robot body position w.r.t world frame
        self.bodyOrn = self.env.body_orn_w.copy()   # initial robot body orientation w.r.t world frame
        self.eePos = self.env.eePos.copy()          # initial robot leg end-effector position w.r.t body frame
        
        self.eeAng = np.array([0., 0., 0., 0., 0., 0.,]) # the deviation of each leg from neutral position, use 0. to initiate a float type array
        self.init_pose = np.zeros((4, 6))
        self.current_pose = np.copy(self.init_pose)

        self.real_robot_control = real_robot_control
        self.xmk, self.imu, self.hexapod, self.fbk_imu, self.fbk_hp, self.group_command, self.group_feedback = self.env.xmk, self.env.imu, self.env.hexapod, self.env.fbk_imu, self.env.fbk_hp, self.env.group_command, self.env.group_feedback
        
        self.trajplanner = TrajPlanner(neutralPos=self.eePos)

        self.max_step_len = 0.2 # maximum stride length in metre
        self.max_rotation = 20 # maximum turn angle in degrees
        # set default value to step parameters, they may change using get_step_params() function
        self.step_len = 0. # stride length in metre
        self.course = 0. # course angle in degrees
        self.rotation = 0. # turn angle in degrees
        self._step_len = np.copy(self.step_len) # record the last step length  # actual step length that the robot will take, may be a smoothed version of cmd_step_len
        self._course = np.copy(self.course) # record the last step course
        self._rotation = np.copy(self.rotation) # record the last step rotation
        self.cmd_step_len = np.copy(self.step_len)  # target step length for the robot to achieve, based on user commands, set by the get_step_params method
        self.cmd_course = np.copy(self.course)
        self.cmd_rotation = np.copy(self.rotation)
        self.cmd_steps = 1 # number of steps to take

        self.traj_dim = self.trajplanner.traj_dim # trajectory dimension for walking and turning, they both have same length
        self.flag = 0 # a flag to record how many steps achieved
        self.is_moving = False # Ture for moving and False for static
        self.smoothing = True # Set to True to enable step smoothing
        self.show_ref_points = show_ref_points # Set to True to show trajectory of legs by plotting reference point in the environment

        self.height_map = self.env.height_map
        self.batch_size = batch_size
        self.opt_vis = opt_vis # visualize final pose obtained from optimzer
        self.optimizer = PamOptimizer(height_map=self.height_map, batch_size=batch_size, vis=opt_vis)
        self.body_interval = body_interval # interval length to move body in world frame (for planner), if initial and final body poses are too far apart

    def step(self, *args, **kwargs):
        '''
        The function to enable Yuna robot step one stride forward, the parameters are get from get_step_params() or manually set
        :param step_len: The step length the robot legs cover during its swing or stance phase in metres, this is measured under robot body frame. The actual step length of first step is halved
        :param course: The robot moving direction, this is measured under robot body frame
        :param rotation: The rotation of robot body per step in radians. The actual rotation of first step is halved
        :param steps: The number of steps the robot will take
        :return: if the step is executed, return True, else return False
        '''
        self.get_step_params(*args, **kwargs)
        cmd_steps = self.cmd_steps
        self.cmd_steps = 1
        
        # if no movement is commanded and robot is stationary at initial position, then no movement
        if self.cmd_step_len == 0.0 and self.cmd_rotation == 0.0 and np.equal(self.current_pose, self.init_pose).all():#
            self.is_moving = False
            return False
        else:
            # change robot status and start moving
            self.is_moving = True
            if self.cmd_step_len == 0 and self.cmd_rotation == 0: # robot executes a single step to stop
                cmd_steps = 1
                self.is_moving = False
            for step in range(int(cmd_steps)):
                for i in range(self.traj_dim):
                    self._smooth_step()
                    traj, end_pose = self.trajplanner.get_loco_traj(self.current_pose, self.step_len, self.course, self.rotation, self.flag, i)
                    self.env.step(traj)
                self.current_pose = end_pose
                self.flag += 1
             
            return True
        
    def goto(self, dx, dy, dtheta):
        '''
        Function to move Yuna robot to a specific position
        :param dx: The x-axis displacement in metres
        :param dy: The y-axis displacement in metres
        :param dtheta: The rotation of robot body orientation in degrees
        '''
        self.smoothing = False # disable step smoothing for more accurate positioning
        cor_coeff_disp = 1.075
        cor_coeff_ang = 1.053
        dx, dy, dtheta = dx*cor_coeff_disp, dy*cor_coeff_disp, dtheta*cor_coeff_ang
        lin_disp = np.sqrt(dx**2 + dy**2)
        ang_disp = np.abs(dtheta)
        lin_steps = np.ceil(lin_disp / self.max_step_len)
        ang_steps = np.ceil(ang_disp / self.max_rotation)
        steps = np.int64(np.max((lin_steps, ang_steps)))

        step_len = lin_disp / steps
        course = np.rad2deg(np.arctan2(dy, dx))
        rotation = np.sign(dtheta) * ang_disp / steps

        for step in range(steps):
            self.step(step_len=step_len, course=course, rotation=rotation)
            course -= rotation
            # if step == 0:
            #     course -= rotation/2
            # else:
            #     course -= rotation
        self.stop()
        self.smoothing = True

    def stop(self):
        '''
        The function to stop Yuna's movements and reset Yuna's pose
        :return: None
        '''
        if self.is_moving:
            self.step(step_len=0,rotation=0)
            self.is_moving = False

    def get_body_matrix(self): # homogeneous transformation matrix of body frame wrt world frame
        rot_mat = self.env.getMatrixFromQuaternion(self.bodyOrn)
        body_mat = np.eye(4)
        body_mat[:3, :3] = rot_mat
        body_mat[:3, 3] = self.bodyPos_w()
        return body_mat
    
    def _eef_world_to_body_frame(self, pos_w, WTB = None):
        if WTB is None:
            WTB = self.get_body_matrix()
        BTW = np.linalg.inv(WTB)
        return np.dot(BTW, np.vstack((pos_w, np.ones((1, 6)))))[0:3, :] # BpE = BTW * WpE
    
    def _eef_body_to_world_frame(self, pos_b, WTB = None):
        if WTB is None:
            WTB = self.get_body_matrix()
        return np.dot(WTB, np.vstack((pos_b, np.ones((1, 6)))))[0:3, :]

    def eePos_b(self): 
        # get end-effector position wrt body frame
        # return self.eePos # calculated
        return self.env.get_leg_pos().copy() # based on pybullet
    
    def eePos_w(self):
        # get end-effector position wrt world frame
        # return self._eef_body_to_world_frame(self.eePos_b()) # calculated
        return self.env.get_leg_pos_in_world_frame().copy() # based on pybullet
    
    def bodyPos_w(self):
        # return self.bodyPos # calculated
        return np.array(self.env.body_pos_w) # based on pybullet
    
    # def swing_leg(self, leg_index, target_pos):
    #     '''
    #     :param leg_index: The index of the leg to move
    #     :param target_pos: The target position of the leg in the world frame
    #     '''
    #     pos_w0 = self.eePos_w().copy()  # initial leg position in world frame
    #     target_leg_midpoint = (pos_w0[:, leg_index] + target_pos) / 2  # use xy midpoint
    #     target_leg_midpoint[2] = max(pos_w0[2, leg_index], target_pos[2]) + STEP_HEIGHT  # raise
    #     pos_w1 = pos_w0.copy()
    #     pos_w1[:, leg_index] = target_leg_midpoint
    #     pos_w2 = pos_w0.copy()
    #     pos_w2[:, leg_index] = target_pos

    #     waypoints = [pos_w1, pos_w2]
    #     # print("waypoints: ", waypoints)
    #     self.move_legs_to_pos_in_world_frame(waypoints)

    def move_legs_to_pos_in_body_frame(self, target_pos_arr):
        '''
        target_pos_arr: a nx3x6 array of n target leg positions wrt BODY frame
        '''
        pos_b0 = self.eePos_b().copy() # self.env.get_leg_pos().copy() # initial leg pos in body frame
        waypoints = [pos_b0] + target_pos_arr
        # print("waypoints: ", waypoints)

        # plan and execute trajectory
        # traj = self.trajplanner.general_traj(waypoints, total_time=len(waypoints) * 0.8)
        traj = self.trajplanner.pam_traj(waypoints, total_time=len(waypoints) * 0.2)

        counter = 0
        for traj_point in traj:
            # print(traj_point)
            self.env.step(traj_point)
            # debug visualization
            if self.show_ref_points and counter % 10 == 0:
                self.env.add_ref_points_wrt_body_frame(traj_point)
                self.env.add_body_frame_ref_point()
            counter += 1
        # print("num of traj points: ", counter)
        self.eePos = target_pos_arr[-1].copy()

    def move_legs_to_pos_in_world_frame(self, target_pos_arr):
        '''
        target_pos_arr: a nx3x6 array of n target leg positions wrt WORLD frame
        '''
        target_pos_body = [self._eef_world_to_body_frame(pos_w) for pos_w in target_pos_arr]
        self.move_legs_to_pos_in_body_frame(target_pos_body)
        
    def move_legs_by_pos_in_body_frame(self, move_by_pos_arr):
        '''
        move_by_pos_arr: a nx3x6 array of n move_by_pos wrt BODY frame, 
                        each column of move_by_pos is the dx dy dz of each leg wrt BODY frame
        '''
        pos_b0 = self.eePos_b().copy() # initial leg pos in body frame
        target_pos_arr = [pos_b0 + move_by_pos for move_by_pos in move_by_pos_arr]
        # print("target_pos_arr: \n", target_pos_arr)
        self.move_legs_to_pos_in_body_frame(target_pos_arr)
    
    def move_legs_by_pos_in_world_frame(self, move_by_pos_arr):
        '''
        move_by_pos_arr: a nx3x6 array of n move_by_pos wrt WORLD frame, 
                        each column of move_by_pos is the dx dy dz of each leg wrt BODY frame
        '''
        pos_w0 = self.eePos_w().copy() # initial leg pos in body frame
        target_pos_arr = [pos_w0 + move_by_pos for move_by_pos in move_by_pos_arr]
        # print("target_pos_arr: \n", target_pos_arr)
        self.move_legs_to_pos_in_world_frame(target_pos_arr)

    def trans_body_by_in_world_frame(self, trans_by_arr):
        '''
        trans_by_arr: a 1x3 array of dx dy dz wrt WORLD frame
        '''
        move_by_pos_arr = -np.tile(trans_by_arr, (6, 1)).T
        self.move_legs_by_pos_in_world_frame([move_by_pos_arr])
        self.bodyPos += trans_by_arr

    def trans_body_to_in_world_frame(self, target_body_pos):
        body_pos_w0 = self.bodyPos_w().copy() # initial body pos in world frame
        trans_by = target_body_pos - body_pos_w0
        self.trans_body_by_in_world_frame(trans_by)

    # def rotx_body(self, angle, num_of_waypoints=2, move=False):
    #     '''
    #     angle: rotation angle in degrees
    #     return: final leg pos wrt body frame to achieve the body rotation
    #     '''
    #     WTB0 = self.get_body_matrix() # initial body frame wrt world
    #     pos_b0 = self.eePos_b() # self.env.get_leg_pos().copy() # initial leg pos in body frame
    #     pos_w0 = self.eePos_w() # np.dot(WTB0, np.vstack((pos_b0, np.ones((1, 6)))))[0:3, :] # initial leg pos in world frame
    #     # print("initial leg pos wrt body frame: \n", pos_b0)
    #     # print("initial body frame wrt world: \n", WTB0)
    #     # print("initial leg pos wrt world frame: \n", pos_w0)
        
    #     target_pos_arr = []
    #     interval = angle / num_of_waypoints
    #     for i in range(1, num_of_waypoints+1):
    #         angle = np.deg2rad(interval*i)
    #         c, s = np.cos(angle), np.sin(angle)
    #         rot_x = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    #         WTB1 = np.dot(WTB0, rot_x) # final body frame wrt world after rotation
    #         # initial and final pos in world frame should be the same --> find final pos in body frame
    #         pos_b1 = np.dot(np.linalg.inv(WTB1), np.vstack((pos_w0, np.ones((1, 6)))))[0:3, :] # final leg pos in body frame
    #         # print("final body frame wrt world: \n", WTB1)
    #         # print("final leg pos wrt body frame: \n", pos_b1)
    #         # print("final leg pos wrt world frame: \n", np.dot(WTB1, np.vstack((pos_b1, np.ones((1, 6)))))[0:3, :])
    #         target_pos_arr.append(pos_b1.copy())
        
        # # debug check (compare with actual)
        # final_actual_pos = self.env.get_leg_pos().copy()
        # final_actual_WTB = self.env.get_body_matrix()
        # print("final actual leg pos wrt body frame: \n", final_actual_pos)
        # print("final actual body frame wrt world: \n", final_actual_WTB)
        # print("final actual leg pos wrt world frame: \n", np.dot(final_actual_WTB, np.vstack((final_actual_pos, np.ones((1, 6)))))[0:3, :])
        
        # if move:
        #     self.move_legs_to_pos_in_body_frame(target_pos_arr)
        # return pos_b1

    # def rotx_trans_body(self, angle, dx, dy, dz, move=False):
    #     pos_b0 = self.env.get_leg_pos().copy() # initial leg pos in body frame
    #     pos_b1 = self.rotx_body(angle) # target pos after rotation in body frame
    #     move_by_pos_arr = (pos_b1 - pos_b0) + self.trans_body(dx, dy, dz, move=False)
    #     if move:
    #         self.move_legs_by_pos_in_body_frame([move_by_pos_arr])
    #     return move_by_pos_arr

    def pam(self, pos, leg_idxs, batch_idx=0):
        initial_body_pos = self.bodyPos_w().copy() # initial body pos in world frame

        self.optimizer.logger.info("----- STARTING PAM -----")
        rot = torch.zeros_like(pos)
        final_pose_start_time = time.time()
        final_params = self.optimizer.solve_multiple_legs_ik(pos=pos, rot=rot, leg_idxs=leg_idxs, has_base_goal=False, plot_filename="waypoint_final.png")
        _, final_base_trans_w, final_leg_trans_w, _ = self.optimizer.get_transformations_from_params(final_params)
        final_pose_end_time = time.time()
        self.optimizer.logger.info(f"Time taken to find final pose: {final_pose_end_time - final_pose_start_time} seconds")
        self.optimizer.logger.info("----- final pose found -----")

        if batch_idx >= self.batch_size:
            print("Error: batch_idx should be less than batch_size. Using batch_idx = 0.")
            batch_idx = 0
        
        final_body_pos_w = final_base_trans_w[batch_idx, :3, 3].cpu().numpy()
        final_eef_pos_w = final_leg_trans_w[batch_idx, :, -1, :3, 3].cpu().numpy().T
        self.optimizer.logger.info(f"final body pos in world frame: {final_body_pos_w}")
        self.optimizer.logger.info(f"final eef pos in world frame: {final_eef_pos_w}")
        if self.opt_vis:
            self.optimizer.visualize(base_trans=final_base_trans_w, leg_trans=final_leg_trans_w, goal=pos)
        
        # Planner
            # To find the final optimized pose, we cannot fix the base_xy:
            #       to allow the optimizer to try different xy body poses in the world frame,
            #       for it to find a final body pose that can reach the goal.
            # Once the final optimized pose is determined, a planner will plan the path
            #       for the hexapod's body based on the initial and final optimized poses.
            # After the path is planned, for each waypoint, make base_xy coincide
            #       with that waypoint origin_xy when calculating the remaining leg poses.
        
        # check distance between intial and final optimized body pose
        dist = np.linalg.norm(final_body_pos_w - initial_body_pos)
        self.optimizer.logger.debug(f"Distance between initial and final body pose: {dist}")

        # interpolate between initial and final optimized body pose
        num_of_waypoints = round(dist / self.body_interval)
        self.optimizer.logger.debug(f"Number of waypoints: {num_of_waypoints}")

        if num_of_waypoints <= 1:
            self.move_to_next_pose(final_body_pos_w, final_eef_pos_w)
            return

        body_waypoints = []
        legs_waypoints = []
        for i in range(1, num_of_waypoints):
            # linear interpolation between initial and final optimized body pose
            body_waypoint = initial_body_pos + i * (final_body_pos_w - initial_body_pos) / num_of_waypoints
            # print(f"Body waypoint {i}: {body_waypoint}")
            
            leg_idxs = []
            temp_pos = torch.tensor([])
            temp_rot = torch.zeros_like(temp_pos)
            
            self.optimizer.logger.info(f"\n--- Solving for waypoint {i} ---")
            start_time = time.time()
            next_params = self.optimizer.solve_multiple_legs_ik(pos=temp_pos, rot=temp_rot, leg_idxs=leg_idxs, has_base_goal=True, target_base_xy=torch.tensor(body_waypoint[:2]), plot_filename=f"waypoint_{i}.png")
            _, next_base_trans_w, next_leg_trans_w, _ = self.optimizer.get_transformations_from_params(next_params)
            end_time = time.time()
            self.optimizer.logger.info(f"Time taken to find waypoint {i}: {end_time - start_time} seconds")
            next_body_pos_w = next_base_trans_w[batch_idx, :3, 3].cpu().numpy()
            next_eef_pos_w = next_leg_trans_w[batch_idx, :, -1, :3, 3].cpu().numpy().T
            self.optimizer.logger.info(f"waypoint {i} body pos in world frame: {next_body_pos_w}")
            self.optimizer.logger.info(f"waypoint {i} eef pos in world frame: {next_eef_pos_w}")
            body_waypoints.append(next_body_pos_w)
            legs_waypoints.append(next_eef_pos_w)
            
        body_waypoints.append(final_body_pos_w)
        legs_waypoints.append(final_eef_pos_w)
        self.optimizer.logger.debug(f"initial body pos: {initial_body_pos}")
        self.optimizer.logger.debug(f"initial eef pos: {self.eePos_w()}")
        self.optimizer.logger.info(f"body waypoints: {body_waypoints}")
        self.optimizer.logger.info(f"legs waypoints: {legs_waypoints}")
        
        self._plot_hexapod_path(body_waypoints, legs_waypoints)
        return body_waypoints, legs_waypoints
        # self.pam_move(body_waypoints, legs_waypoints)

    def pam_move(self, body_waypoints, legs_waypoints, motion=("press_button", 1)):
        # motion: ("normal_walk", None) or ("press_button", leg_idx)
        num_of_waypoints = len(body_waypoints)
        # normal walking
        for i in range(num_of_waypoints-1):
            self.move_to_next_pose_tripod_gait(body_waypoints[i], legs_waypoints[i])

        if motion[0] == "normal_walk":
            self.move_to_next_pose_tripod_gait(body_waypoints[-1], legs_waypoints[-1])

        if motion[0] == "press_button":
            self._pam_press_button_motion(body_waypoints[-1], legs_waypoints[-1], motion[1])

    def pam_press_button(self, button_pos, leg_idx):
        body_waypoints, legs_waypoints = self.pam(button_pos, [leg_idx])
        self.pam_move(body_waypoints, legs_waypoints, ("press_button", leg_idx))

    def _pam_press_button_motion(self, target_body_pos_w, target_eef_pos_w, press_leg_idx=1):
        # assuming can only press with front two legs (index 0 or 1)      
        initial_body_pos_w = self.bodyPos_w().copy() # initial body pos in world frame
        # print("initial body pos: ", initial_body_pos_w)
        # print("target body pos: ", target_body_pos_w)
        # print("initial eef pos: ", self.eePos_w())
        # print("target eef pos: ", target_eef_pos_w)        
        for leg_idx in range(NUM_LEGS):
            if leg_idx != press_leg_idx:
                # pre processing, force adjust eef pos to be at least at the height of the next position 
                # so that it doesn't push too much on the ground / too high and not touching the gorund
                height_at_next_pos = self.height_map.get_height_at(target_eef_pos_w[0, leg_idx], target_eef_pos_w[1, leg_idx])
                target_eef_pos_w[2, leg_idx] = height_at_next_pos
        
        body_keyframe_w, leg_keyframe_w = self.trajplanner.pam_press_button_keyframe(initial_body_pos_w, target_body_pos_w, self.eePos_w(), target_eef_pos_w, press_leg_idx)
        leg_keyframe_b = self.keyframe_b_from_w(body_keyframe_w, leg_keyframe_w)
        self.move_legs_to_pos_in_body_frame(leg_keyframe_b)

    def move_to_next_pose_tripod_gait(self, next_body_pos_w, next_eef_pos_w):
        # ASSUMING ALWAYS ALL SIX LEGS ON GROUND WHEN MOVING WITH TRIPOD GAIT
        initial_body_pos_w = self.bodyPos_w().copy() # initial body pos in world frame
        
        for leg_idx in range(NUM_LEGS):
            # pre processing, force adjust eef pos to be at least at the height of the next position 
            # so that it doesn't push too much on the ground / too high and not touching the gorund
            height_at_next_pos = self.height_map.get_height_at(next_eef_pos_w[0, leg_idx], next_eef_pos_w[1, leg_idx])
            next_eef_pos_w[2, leg_idx] = height_at_next_pos

        body_keyframe_w, leg_keyframe_w = self.trajplanner.pam_tripod_keyframe(initial_body_pos_w, next_body_pos_w, self.eePos_w(), next_eef_pos_w)
        leg_keyframe_b = self.keyframe_b_from_w(body_keyframe_w, leg_keyframe_w)
        self.move_legs_to_pos_in_body_frame(leg_keyframe_b)

    def move_to_next_pose_wave_gait(self, next_body_pos_w, next_eef_pos_w, leg_sequence=[0, 1, 2, 3, 4, 5]):
        # MAINLY USED TO MOVE TO FINAL POSE (LAST STEP)
        initial_body_pos_w = self.bodyPos_w().copy() # initial body pos in world frame
        support_legs = [True] * 6
        
        for leg_idx in range(NUM_LEGS):
            height_at_next_pos = self.height_map.get_height_at(next_eef_pos_w[0, leg_idx], next_eef_pos_w[1, leg_idx])
            if next_eef_pos_w[2, leg_idx] - height_at_next_pos > RAISE_THRES: # skip those which legs are intentionally raised
                support_legs[leg_idx] = False
            else:
                next_eef_pos_w[2, leg_idx] = height_at_next_pos
        body_keyframe_w, leg_keyframe_w = self.trajplanner.pam_wave_keyframe(initial_body_pos_w, next_body_pos_w, self.eePos_w(), next_eef_pos_w, support_legs=support_legs, leg_sequence=leg_sequence)
        leg_keyframe_b = self.keyframe_b_to_w(body_keyframe_w, leg_keyframe_w)
        self.move_legs_to_pos_in_body_frame(leg_keyframe_b)

    def keyframe_b_from_w(self, body_keyframe_w, leg_keyframe_w):
        # TODO: if add rotation need include rotation matrix
        leg_keyframe_b = []
        keyframe_len = len(body_keyframe_w)
        for i in range(1, keyframe_len):
            # rot_mat = self.env.getMatrixFromQuaternion(self.bodyOrn)
            wTb = np.eye(4) # homogeneous transformation matrix of body frame wrt world frame
            # wTb[:3, :3] = rot_mat
            wTb[:3, 3] = body_keyframe_w[i]
            leg_pos_w = leg_keyframe_w[i]
            leg_keyframe_b.append(self._eef_world_to_body_frame(leg_pos_w, wTb))
        # print("leg_keyframe_b: ", leg_keyframe_b)
        return leg_keyframe_b
    

    def _plot_hexapod_path(self, body_waypoints, legs_waypoints):
        cmap = plt.get_cmap('tab10')
        num_points = len(body_waypoints)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(num_points):
            body_wp = body_waypoints[i]
            leg_waypoints = legs_waypoints[i]
            color = cmap(i / num_points)
            
            # Plot body waypoint
            ax.scatter(body_wp[0], body_wp[1], body_wp[2], c=[color], marker='o', label=f'Waypoint {i+1}')
            
            # Plot leg waypoints
            for j in range(leg_waypoints.shape[1]):
                ax.scatter(leg_waypoints[0, j], leg_waypoints[1, j], leg_waypoints[2, j], c=[color], marker='^')
                ax.plot([body_wp[0], leg_waypoints[0, j]],
                        [body_wp[1], leg_waypoints[1, j]],
                        [body_wp[2], leg_waypoints[2, j]], color=color, linestyle='--')
            # Draw hexagon connecting the leg waypoints in the specified sequence
            sequence = [0, 1, 3, 5, 4, 2, 0]  # Indices for legs 1, 2, 4, 6, 5, 3
            for k in range(len(sequence) - 1):
                ax.plot([leg_waypoints[0, sequence[k]], leg_waypoints[0, sequence[k + 1]]],
                        [leg_waypoints[1, sequence[k]], leg_waypoints[1, sequence[k + 1]]],
                        [leg_waypoints[2, sequence[k]], leg_waypoints[2, sequence[k + 1]]], color=color, linestyle='-')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()


    def disconnect(self):
        '''
        Disable real robot motors, disconnect from pybullet environment and exit the programme
        :return: None
        '''
        self.env.close()

    def get_step_params(self, *args, **kwargs):
        '''
        This function is used to listen to the commands from the user
        :param step_len: The step length the robot legs cover during its swing or stance phase, this is measured under robot body frame. The actual step length of first step is halved
        :param course: The robot moving direction, this is measured under robot body frame
        :param rotation: The rotation of robot body per step. The actual rotation of first step is halved
        :param steps: The number of steps the robot will take
        :return: None
        '''
        if len(args) > 0 or len(kwargs) > 0:# if there is any input command
            if len(args) + len(kwargs) > 4:
                raise ValueError('Expected at most 4 arguments: step_len, course, rotation and steps, got %d' % (len(args) + len(kwargs)))
            # set default values
            legal_keys = ['step_len', 'course', 'rotation', 'steps']
            default_values = [0., 0., 0., 1]
            for key in legal_keys:
                setattr(self, key, default_values[legal_keys.index(key)])
            # combine the args and kwargs, if there is a value in args not stated in kwargs, it will be assigned to the first key in legal_keys that is not in kwargs
            for value in args:
                for key in legal_keys:
                    if key not in kwargs:
                        kwargs[key] = value
                        break
            # check the input commands and set input parameters as attributes
            for key, value in kwargs.items():
                if key not in legal_keys:
                    raise TypeError('Invalid keyword argument: {}'.format(key))
                # pre-processing of the input commands
                if key == 'step_len':
                    value = np.clip(value, -self.max_step_len, self.max_step_len)
                if key == 'course':
                    value = np.deg2rad(value)
                if key == 'rotation':
                    value = np.clip(value, -self.max_rotation, self.max_rotation)
                    value = np.deg2rad(value)
                if key == 'steps':
                    value = np.ceil(np.abs(value))
                setattr(self, 'cmd_' + key, value)

    def _smooth_step(self):
        '''
        This function is used to smooth the robot's movements to avoid abrupt changes in robot's legs' task coordinate pose
        :return: None
        '''
        rho = 0.05 # soft copy rate (A smaller rho value smoother transitions, a larger value more immediate transitions)
        if self.smoothing:
            # calculate diff in position
            _pos = transxy((0.,0.), self._step_len, self._course)
            cmd_pos = transxy((0.,0.), self.cmd_step_len, self.cmd_course)
            dpos = np.linalg.norm(cmd_pos - _pos)
            
            # calculate diff in rotation
            _rot = self._rotation
            cmd_rot = self.cmd_rotation
            drot = np.rad2deg(np.abs(cmd_rot - _rot))
            
            # if the robot's current movement is close enough to the command movement, and command step length and rotation are zero
            if dpos < 2 * self.max_step_len / 10  and drot < 2 * self.max_rotation / 10 and self.cmd_step_len == 0 and self.cmd_rotation == 0:
                rho = 1
                
            # apply smoothing
            pos = rho * cmd_pos + (1 - rho) * _pos
            self.step_len = np.sqrt(pos[0]**2 + pos[1]**2)
            self.course = np.arctan2(pos[1], pos[0])
            self.rotation = rho * self.cmd_rotation + (1 - rho) * self._rotation

            self._step_len = np.copy(self.step_len)
            self._course = np.copy(self.course)
            self._rotation = np.copy(self.rotation)
        else:
            self.step_len = self.cmd_step_len
            self.course = self.cmd_course
            self.rotation = self.cmd_rotation
    
    def _get_current_pos(self):
        '''
        Get the current end effector positions of all 6 legs from the current pose
        :return: current leg end effector positions based on the current pose of each leg's task coordinate frame with respect to the robot body frame
        '''
        current_pos = np.zeros((3, 6))
        for leg_index in range(6):
            current_pos[:, leg_index] = self.trajplanner.pose2pos(self.current_pose[:, leg_index], leg_index)
        return current_pos

    def print_robot_info(self):
        print("yuna.bodyPos: \n", yuna.bodyPos_w())
        print("yuna.eePos_b: \n", yuna.eePos_b())
        print("yuna.eePos_b2: \n",yuna._eef_world_to_body_frame(yuna.eePos_w()))
        print("yuna.eePos_w: \n", yuna.eePos_w())
        print()
        print("yuna.env.bodyPos: \n", yuna.env.body_pos_w)
        print("yuna.env.eePos_b: \n", yuna.env.eePos)
        print("yuna.env.eePos_w: \n", yuna.env.get_leg_pos_in_world_frame())
        print()
        print("body diff: \n", yuna.bodyPos_w() - yuna.env.body_pos_w)
        print("eef_b diff: \n", yuna.eePos_b() - yuna.env.eePos)
        print("eef_w diff: \n", yuna.eePos_w() - yuna.env.get_leg_pos_in_world_frame())
        print("-" * 50)
        print()

if __name__ == '__main__':
    # motion test of yuna robot
    import time
    yuna = Yuna(real_robot_control=0, pybullet_on=1)
    yuna.env.camerafollow = False

    # -- check initial values ---
    print("---initial---")
    yuna.print_robot_info()

    # yuna.step(step_len=0.1, course=0, rotation=0, steps=10)

    # -- test move_legs_to_pos_in_body_frame ---
    # target_eePos_r = [np.array([[ 0.4901,  0.49,    0.0426,  0.0424, -0.4475, -0.4476],
    #                         [ 0.2338, -0.234,   0.4914, -0.4914,  0.3076, -0.3074],
    #                         [      0, -0.1252, -0.1252, -0.1252,  -0.1252, -0.1252]])]
    # yuna.move_legs_to_pos_in_body_frame(target_eePos_r)
    # print("---final---")
    # yuna.print_robot_info()

    # -- test move_legs_to_pos_in_world_frame ---
    target_eePos_w = [np.array([[ 0.4919,  0.4919,  0.0442,  0.0442, -0.4467, -0.4465],
                            [ 0.2335, -0.2349,  0.4915, -0.4924,  0.3073, -0.3082],
                            [ 0.2224,  0.0228,  0.0293,  0.0295,  0.0252,  0.0254]])]
    yuna.move_legs_to_pos_in_world_frame(target_eePos_w)
    print("---final---")
    yuna.print_robot_info()

    # -- test move_legs_by_pos_in_body_frame ---
    # target_eePos_diff_w = [np.array([[0, 0, 0, 0, 0, 0],
    #                                  [0, 0, 0, 0, 0, 0],
    #                                  [0.2, 0, 0, 0, 0, 0.2]])]
    # yuna.move_legs_by_pos_in_body_frame(target_eePos_diff_w)
    # print("---final---")
    # yuna.print_robot_info()

    # -- test move_legs_by_pos_in_world_frame ---
    # target_eePos_diff_w = [np.array([[0, 0, 0, 0, 0, 0],
    #                                  [0, 0, 0, 0, 0, 0],
    #                                  [-0.2, 0, 0, 0, 0, -0.2]])]
    # yuna.move_legs_by_pos_in_world_frame(target_eePos_diff_w)
    # print("---final---")
    # yuna.print_robot_info()

    # -- test trans_body_by_in_world_frame ---
    trans_body_by = np.array([0.05, 0, 0.05])
    yuna.trans_body_by_in_world_frame(trans_body_by)
    print("---final---")
    yuna.print_robot_info()

    # -- test trans_body_to_in_world_frame ---
    # trans_body_to = np.array([0.05, 0, 0.145])
    # yuna.trans_body_to_in_world_frame(trans_body_to)
    # print("---final---")
    # yuna.print_robot_info()

    # print('There will be a series of robot movements with randomly generated parameters')
    # for motion in range(10):
    #     step_len = np.random.uniform(0, yuna.max_step_len)
    #     course = np.random.rand() * 360
    #     rotation = np.random.uniform(-yuna.max_rotation, yuna.max_rotation)
    #     steps = np.random.randint(1, 6)
    #     print('Yuna command summary: step length = ' + str(np.around(step_len, decimals=4)) \
    #             + ', course direction = ' + str(np.around(course, decimals=2)) \
    #             + ', rotation angle = ' + str(np.around(rotation, decimals=2)) \
    #             + ', step number = ' + str(np.around(steps, decimals=2)))
    #     yuna.step(step_len=step_len, course=course, rotation=rotation, steps=steps)
    #     if np.random.rand() > 0.7:
    #         yuna.stop()
    #         time.sleep(1)
    #         print('Yuna stopped')

    time.sleep(60)
    yuna.disconnect()

