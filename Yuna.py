from Yuna_TrajPlanner import TrajPlanner
from Yuna_Env import YunaEnv
import numpy as np
from functions import solveFK

class Yuna:
    def __init__(self, visualiser=True, camerafollow=True, real_robot_control=False, pybullet_on=True):
        self.env = YunaEnv(visualiser=visualiser, camerafollow=camerafollow, real_robot_control=real_robot_control, pybullet_on=pybullet_on)
        self.eePos = self.env.eePos.copy()
        self.eeAng = np.array([0., 0., 0., 0., 0., 0.,]) # the diviation of each leg from neutral position, use 0. to initiate a float type array
        self.current_pose = np.zeros((4, 6))

        self.real_robobt_control = real_robot_control
        self.xmk, self.imu, self.hexapod, self.fbk_imu, self.fbk_hp, self.group_command, self.group_feedback = self.env.xmk, self.env.imu, self.env.hexapod, self.env.fbk_imu, self.env.fbk_hp, self.env.group_command, self.env.group_feedback
        
        self.trajplanner = TrajPlanner(neutralPos=self.eePos)

        self.max_step_len = 0.2 # maximum stride length in metre
        self.max_rotation = 20 # maximum turn angle in degrees

        self.traj_dim = self.trajplanner.traj_dim # trajectory dimension for walking and turning, they both have same lenghth
        self.flag = 0 # a flag to record how many steps achieved
        self.is_moving = False # Ture for moving and False for static

    def step(self, step_len=0.2, course=0, rotation=0, steps=1):
        '''
        The function to enable Yuna robot step one stride forward

        :param step_len: The step length the robot legs cover during its swing or stance phase, this is measured under robot body frame. The actual step length of first step is halved
        :param course: The robot moving direction, this is measured under robot body frame
        :param rotation: The rotation of robot body per step. The actual rotation of first step is halved
        :return: None
        '''
        self.is_moving = True
        # pre-processing of the input commands
        step_len = np.clip(step_len, -self.max_step_len, self.max_step_len)
        rotation = np.clip(rotation, -self.max_rotation, self.max_rotation)
        course = np.deg2rad(course)
        rotation = np.deg2rad(rotation)

        for step in range(steps):
            traj, end_pose = self.trajplanner.get_loco_traj(self.current_pose, step_len, course, rotation, self.flag)
            for i in range(self.traj_dim):
                self.env.step(traj[i])
            self.current_pose = end_pose
            self.flag += 1

    def stop(self):
        '''
        The function to stop Yuna's movements and reset Yuna's pose
        '''
        if self.is_moving:
            self.step(0, 0, 0)
            self.is_moving = False
    
    def disconnect(self):
        '''
        Disable real robot motors, disconnect from pybullet environment and exit the programme
        '''
        self.env.close()
    
    def _get_current_pos(self):
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
        yuna.step(step_len, course, rotation, steps)
        if np.random.rand() > 0.7:
            yuna.stop()
            time.sleep(1)
            print('Yuna stopped')

    time.sleep(60)
    yuna.disconnect()

