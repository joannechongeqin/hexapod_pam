from Yuna_TrajPlanner import TrajPlanner
from Yuna_Env import YunaEnv
import numpy as np
from functions import transxy, solveFK, rotx
import time

class Yuna:
    def __init__(self, visualiser=True, camerafollow=True, real_robot_control=False, pybullet_on=True, show_ref_points=False):
        # initialise the environment
        self.env = YunaEnv(visualiser=visualiser, camerafollow=camerafollow, real_robot_control=real_robot_control, pybullet_on=pybullet_on)
        self.eePos = self.env.eePos.copy() # robot leg end-effecter position w.r.t body frame
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
    
    def move_legs_to_pos_in_body_frame(self, target_pos_arr):
        '''
        target_pos_arr: an array of 3x6 arrays of target leg positions wrt BODY frame
        '''
        pos_b0 = self.env.get_leg_pos().copy() # initial leg pos in body frame
        waypoints = [pos_b0] + target_pos_arr
        
        # plan and execute trajectory
        traj = self.trajplanner.general_traj(waypoints, total_time=len(waypoints) * 0.8)
        counter = 0
        for traj_point in traj:
            self.env.step(traj_point)
            # debug visualization
            if self.show_ref_points and counter % 10 == 0:
                self.env.add_ref_points_wrt_body_frame(traj_point)
                self.env.add_body_frame_ref_point()
            counter += 1
        print("num of traj points: ", counter) 
    
    def move_legs_to_pos_in_world_frame(self, target_pos_arr):
        '''
        target_pos_arr: an array of 3x6 arrays of target leg positions wrt WORLD frame
        '''
        WTB = self.env.get_body_matrix() # body frame wrt world
        BTW = np.linalg.inv(WTB)
        target_pos_body = [np.dot(BTW, np.vstack((pos, np.ones((1, 6)))))[0:3, :] for pos in target_pos_arr] # BpE = BTW * WpE
        self.move_legs_to_pos_in_body_frame(target_pos_body)
        
    def move_legs_by_pos_in_body_frame(self, move_by_pos_arr):
        '''
        move_by_pos_arr: an array of 3x6 arrays (move_by_pos), each column of move_by_pos is the dx dy dz of each leg wrt BODY frame
        '''
        pos_b = self.env.get_leg_pos().copy() # leg pos in body frame
        target_pos_arr = []
        for move_by_pos in move_by_pos_arr:
            pos_b += move_by_pos
            target_pos_arr.append(pos_b.copy())
        self.move_legs_to_pos_in_body_frame(target_pos_arr)
    
    def move_legs_by_pos_in_world_frame(self, move_by_pos_arr):
        '''
        move_by_pos_arr: an array of 3x6 arrays (move_by_pos), each column of move_by_pos is the dx dy dz of each leg wrt WORLD frame
        '''
        pos_b = self.env.get_leg_pos().copy() # leg pos in body frame
        WTB = self.env.get_body_matrix() # body frame wrt world
        BTW = np.linalg.inv(WTB)
        pos_w = np.dot(WTB, np.vstack((pos_b, np.ones((1, 6)))))[0:3, :] # leg pos in world frame
        target_pos_arr = []
        for move_by_pos in move_by_pos_arr:
            pos_w += move_by_pos
            target_pos_arr.append(pos_w.copy())
        self.move_legs_to_pos_in_world_frame(target_pos_arr)
        
    def rotx_body(self, angle, num_of_waypoints=2, move=False):
        '''
        angle: rotation angle in degrees
        return: final leg pos wrt body frame to achieve the body rotation
        '''
        WTB0 = self.env.get_body_matrix() # initial body frame wrt world
        pos_b0 = self.env.get_leg_pos().copy() # initial leg pos in body frame
        pos_w0 = np.dot(WTB0, np.vstack((pos_b0, np.ones((1, 6)))))[0:3, :] # initial leg pos in world frame
        # print("initial leg pos wrt body frame: \n", pos_b0)
        # print("initial body frame wrt world: \n", WTB0)
        # print("initial leg pos wrt world frame: \n", pos_w0)
        
        target_pos_arr = []
        interval = angle / num_of_waypoints
        for i in range(1, num_of_waypoints+1):
            angle = np.deg2rad(interval*i)
            c, s = np.cos(angle), np.sin(angle)
            rot_x = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
            WTB1 = np.dot(WTB0, rot_x) # final body frame wrt world after rotation
            # initial and final pos in world frame should be the same --> find final pos in body frame
            pos_b1 = np.dot(np.linalg.inv(WTB1), np.vstack((pos_w0, np.ones((1, 6)))))[0:3, :] # final leg pos in body frame
            # print("final body frame wrt world: \n", WTB1)
            # print("final leg pos wrt body frame: \n", pos_b1)
            # print("final leg pos wrt world frame: \n", np.dot(WTB1, np.vstack((pos_b1, np.ones((1, 6)))))[0:3, :])
            target_pos_arr.append(pos_b1.copy())
        
        # # debug check (compare with actual)
        # final_actual_pos = self.env.get_leg_pos().copy()
        # final_actual_WTB = self.env.get_body_matrix()
        # print("final actual leg pos wrt body frame: \n", final_actual_pos)
        # print("final actual body frame wrt world: \n", final_actual_WTB)
        # print("final actual leg pos wrt world frame: \n", np.dot(final_actual_WTB, np.vstack((final_actual_pos, np.ones((1, 6)))))[0:3, :])
        
        if move:
            self.move_legs_to_pos_in_body_frame(target_pos_arr)
        return pos_b1
    
    def trans_body(self, dx, dy, dz, move=False):
        move_by_pos_arr = np.array([[-dx] * 6, [-dy] * 6, [-dz] * 6])  
        if move:       
            self.move_legs_by_pos_in_body_frame([move_by_pos_arr])
        return move_by_pos_arr
    
    def trans_body_in_world_frame(self, dx, dy, dz, move=False):
        move_by_pos_arr = np.array([[-dx] * 6, [-dy] * 6, [-dz] * 6])
        if move:
            self.move_legs_by_pos_in_world_frame([move_by_pos_arr])
        # return

    def rotx_trans_body(self, angle, dx, dy, dz, move=False):
        pos_b0 = self.env.get_leg_pos().copy() # initial leg pos in body frame
        pos_b1 = self.rotx_body(angle) # target pos after rotation in body frame
        move_by_pos_arr = (pos_b1 - pos_b0) + self.trans_body(dx, dy, dz, move=False)
        if move:
            self.move_legs_by_pos_in_body_frame([move_by_pos_arr])
        return move_by_pos_arr
    
    # def wall_transition_step_ground_leg(self, step_len, leg4_step_half=False, raise_h=0.05):
    #     '''
    #     assuming stepping sideway to left, ground legs are at the right side
    #     '''
    #     for leg_index in [1, 3, 5]:
    #         if leg4_step_half and leg_index == 3:
    #             final_step_len = step_len / 2
    #         else:
    #             final_step_len = step_len
    #         raise_leg = np.zeros((3,6))
    #         raise_leg[:, leg_index] = [0, final_step_len/2, raise_h]
    #         step_leg = np.zeros((3,6))
    #         step_leg[:, leg_index] = [0, final_step_len/2, -raise_h]
    #         self.move_legs_by_pos_in_world_frame([raise_leg, step_leg])
    
    # def wall_transition_first_step_wall_leg(self, step_height, wall_dist):
    #     '''
    #     :param step_height: The height of the first step
    #     :param wall_dist: The distance of the first step to the wall [leg1, leg3, leg5]
    #     assuming stepping sideway to left, wall legs are at the left side
    #     '''
    #     for leg_index in [0, 2, 4]:
    #         raise_leg = np.zeros((3,6))
    #         raise_leg[:, leg_index] = [0, 0, step_height]
    #         step_leg = np.zeros((3,6))
    #         step_leg[:, leg_index] = [0, wall_dist, 0]
    #         self.move_legs_by_pos_in_world_frame([raise_leg, step_leg])
    
    # def wall_transition_step_wall_leg(self, step_len, clearance=0.06):
    #     for leg_index in [0, 2, 4]:
    #         raise_leg = np.zeros((3,6))
    #         raise_leg[:, leg_index] = [0, -clearance/2, step_len/2]
    #         step_leg = np.zeros((3,6))
    #         step_leg[:, leg_index] = [0, +clearance/2, step_len/2]
    #         self.move_legs_by_pos_in_world_frame([raise_leg, step_leg])

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

if __name__ == '__main__':
    # motion test of yuna robot
    import time
    yuna = Yuna()
    print('There will be a series of robot movements with randomly generated parameters')
    for motion in range(10):
        step_len = np.random.uniform(0, yuna.max_step_len)
        course = np.random.rand() * 360
        rotation = np.random.uniform(-yuna.max_rotation, yuna.max_rotation)
        steps = np.random.randint(1, 6)
        print('Yuna command summary: step length = ' + str(np.around(step_len, decimals=4)) \
                + ', course direction = ' + str(np.around(course, decimals=2)) \
                + ', rotation angle = ' + str(np.around(rotation, decimals=2)) \
                + ', step number = ' + str(np.around(steps, decimals=2)))
        yuna.step(step_len=step_len, course=course, rotation=rotation, steps=steps)
        if np.random.rand() > 0.7:
            yuna.stop()
            time.sleep(1)
            print('Yuna stopped')

    time.sleep(60)
    yuna.disconnect()

