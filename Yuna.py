from Yuna_TrajPlanner import TrajPlanner
from Yuna_Env import YunaEnv
import numpy as np
from functions import trans, solveFK, rotx
import time

class Yuna:
    def __init__(self, visualiser=True, camerafollow=True, real_robot_control=False, pybullet_on=True):
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
            

    def move_legs(self, move_by_pos_arr):
        '''
        move_by_pos_arr: an array of 3x6 arrays (move_by_pos), each column of move_by_pos is the dx dy dz of each leg
        :return: None
        '''
        pos = self.env.get_leg_pos().copy()
        waypoints = [pos]

        for move_by_pos in move_by_pos_arr:
            pos = pos.copy() + move_by_pos
            waypoints.append(pos)
        
        # plan and execute trajectory
        traj = self.trajplanner.general_traj(waypoints, total_time=3)
        print(traj.shape, traj)
        for traj_point in traj:
            self.env.step(traj_point)
        # update eePos?
    
    def rotx_body(self, angle):
        '''
        angle: rotation angle in degrees
        to rotate robot body by angle is equals to rotate robot legs by -angle wrt base
        '''
        angle_rad = np.deg2rad(-angle)
        body_pos, body_orn = self.env.get_body_pose()
        legs_pos = self.env.get_leg_pos().copy()
        print(body_pos, body_orn, legs_pos)
        new_leg_pos = np.zeros((3, 6))
        for leg_index in range(6):
            leg_pos = legs_pos[:, leg_index]
            new_pos = rotx(pos=leg_pos, angle=-angle_rad, pivot=body_pos)
            new_leg_pos[:, leg_index] = new_pos
        move_by_pos = new_leg_pos - legs_pos
        self.move_legs([move_by_pos])
        
  
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
            _pos = trans((0.,0.), self._step_len, self._course)
            cmd_pos = trans((0.,0.), self.cmd_step_len, self.cmd_course)
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

