from Yuna_TrajPlanner import TrajPlanner
from Yuna_env import YunaEnv
import numpy as np

class Yuna:
    def __init__(self, visualiser=True, camerafollow=True, real_robot_control=False):
        self.trajplanner = TrajPlanner()
        self.env = YunaEnv(visualiser=visualiser, camerafollow=camerafollow, real_robot_control=real_robot_control)
        self.real_robobt_control = real_robot_control
        self.xmk, self.imu, self.hexapod, self.fbk_imu, self.fbk_hp, self.group_command, self.group_feedback = self.env.xmk, self.env.imu, self.env.hexapod, self.env.fbk_imu, self.env.fbk_hp, self.env.group_command, self.env.group_feedback
        self.maxstride = 0.3 # maximum stride length in metre
        self.maxturn = 20 # maximum turn angle in degrees
        self.eePos = self.env.eePos.copy()
        self.current_pos = self.eePos.copy()
        self.end_pos = self.eePos.copy()
        self.eeAng = np.array([0., 0., 0., 0., 0., 0.,])#the diviation of each leg from neutral position, use 0. to initiate a float type array
        self.current_ang = self.eeAng.copy()
        self.end_ang = self.current_ang.copy()
        self.traj_dim = self.trajplanner.swing_dim # trajectory dimension for walking and turning, both have same lenghth
        self.walk_flag = 0 # a flag to record how many steps achieved in walking
        self.turn_flag = 0 # a flag to record how many steps achieved in turning
        self.tripod1 = [0 ,3, 4] # leg index for leg 1, 4, 5
        self.tripod2 = [1, 2, 5] # leg index for leg 2, 3, 6
        self.mode = 'none'

    def walk(self, stride, angle=0, step=1):
        if self.mode == 'turn':
            self.stop()
        stride = np.clip(stride, -self.maxstride, self.maxstride)
        angle = np.radians(angle)
        self.mode = 'walk'
        for i in range(step):
            traj = self._get_walk_traj(stride, angle)
            for i in range(self.traj_dim):
                self.env.step(traj[i])
            self.current_pos = traj[i].copy()
            self.walk_flag += 1

    def turn(self, deg_per_step, step=1):
        if self.mode == 'stop':
            self.stop()
        deg_per_step = np.clip(deg_per_step, -self.maxturn, self.maxturn)
        angle = np.radians(deg_per_step)
        self.mode = 'turn'
        for i in range(step):
            traj = self._get_turn_traj(angle)
            for i in range(self.traj_dim):
                self.env.step(traj[i])
            self.current_ang = self.end_ang.copy()
            self.turn_flag += 1

    def stop(self):
        if self.mode != 'none':
            if self.mode == 'walk':
                self.walk(0)
            else:
                self.turn(0)
            self.mode = 'none'

    def disconnect(self):
        self.env.close()
    
    def _get_walk_traj(self, stride, angle):
        traj = np.zeros((self.traj_dim, 3, 6))
        self._get_walk_traget_pos(stride, angle)
        if self.walk_flag % 2 == 0:
            for leg_index in self.tripod1:
                traj[:,:,leg_index] = self.trajplanner.walk_swing_traj(init_pos=self.current_pos[:, leg_index], end_pos=self.end_pos[:, leg_index])
            for leg_index in self.tripod2:
                traj[:,:,leg_index] = self.trajplanner.walk_support_traj(init_pos=self.current_pos[:, leg_index], end_pos=self.end_pos[:, leg_index])
        else:
            for leg_index in self.tripod1:
                traj[:,:,leg_index] = self.trajplanner.walk_support_traj(init_pos=self.current_pos[:, leg_index], end_pos=self.end_pos[:, leg_index])
            for leg_index in self.tripod2:
                traj[:,:,leg_index] = self.trajplanner.walk_swing_traj(init_pos=self.current_pos[:, leg_index], end_pos=self.end_pos[:, leg_index])
        return traj

    def _get_turn_traj(self, angle):
        traj = np.zeros((self.traj_dim, 3, 6))
        self._get_turn_traget_ang(angle)
        if self.turn_flag % 2 ==0:
            for leg_index in self.tripod1:
                traj[:,:,leg_index] = self.trajplanner.turn_swing_traj(neutral_pos=self.eePos[:,leg_index], init_ang=self.current_ang[leg_index], end_ang=self.end_ang[leg_index])
            for leg_index in self.tripod2:
                traj[:,:,leg_index] = self.trajplanner.turn_support_traj(neutral_pos=self.eePos[:,leg_index], init_ang=self.current_ang[leg_index], end_ang=self.end_ang[leg_index])
        else:
            for leg_index in self.tripod1:
                traj[:,:,leg_index] = self.trajplanner.turn_support_traj(neutral_pos=self.eePos[:,leg_index], init_ang=self.current_ang[leg_index], end_ang=self.end_ang[leg_index])
            for leg_index in self.tripod2:
                traj[:,:,leg_index] = self.trajplanner.turn_swing_traj(neutral_pos=self.eePos[:,leg_index], init_ang=self.current_ang[leg_index], end_ang=self.end_ang[leg_index])
        return traj

    def _get_walk_traget_pos(self, stride, angle):
        if stride == 0:
            self.end_pos = self.eePos.copy()
        else:
            if self.walk_flag % 2 == 0:
                for leg_index in self.tripod1:
                    self.end_pos[:, leg_index] = self._trans(self.eePos[:,leg_index], stride/2, angle)
                for leg_index in self.tripod2:
                    self.end_pos[:, leg_index] = self._trans(self.eePos[:,leg_index], stride/2, angle + np.pi)
            else:
                for leg_index in self.tripod1:
                    self.end_pos[:, leg_index] = self._trans(self.eePos[:,leg_index], stride/2, angle + np.pi)
                for leg_index in self.tripod2:
                    self.end_pos[:, leg_index] = self._trans(self.eePos[:,leg_index], stride/2, angle)
    
    def _get_turn_traget_ang(self, angle):
        if angle == 0:
            self.end_ang = self.eeAng.copy()
        else:
            if self.turn_flag % 2 == 0:
                for leg_index in self.tripod1:
                    self.end_ang[leg_index] = self.eeAng[leg_index] + angle / 2
                for leg_index in self.tripod2:
                    self.end_ang[leg_index] = self.eeAng[leg_index] - angle / 2
            else:
                for leg_index in self.tripod1:
                    self.end_ang[leg_index] = self.eeAng[leg_index] - angle / 2
                for leg_index in self.tripod2:
                    self.end_ang[leg_index] = self.eeAng[leg_index] + angle / 2

    def _rot(self, pos, angle):
        c, s = np.cos(angle), np.sin(angle)
        rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pos_ = np.matmul(rot_z, pos)
        return pos_
    
    def _trans(self, pos, distance, angle):
        pos_ = pos.copy()
        pos_[0] += distance * np.cos(angle)
        pos_[1] += distance * np.sin(angle)
        return pos_

if __name__ == '__main__':
    # test motions
    yuna = Yuna()

    yuna.walk(stride=-0.4, angle=0, step=3)#math.pi/2
    yuna.stop()
    yuna.walk(stride=0.05, angle=0,step=5)
    yuna.walk(stride=0.3, angle=0,step=5)
    yuna.walk(stride=0.1, angle=180,step=8)
    yuna.walk(stride=0.1, angle=60,step=8)
    yuna.walk(stride=0.1, angle=120,step=8)
    yuna.walk(stride=0.1, angle=270,step=8)
    yuna.stop()
    yuna.turn(deg_per_step=30, step=6)
    yuna.turn(deg_per_step=15, step=5)
    yuna.walk(stride=0.1, angle=270,step=8)
    yuna.turn(deg_per_step=-30, step=8)
    yuna.turn(deg_per_step=5, step=8)
    yuna.stop()

''' def locomotion2manipulation(self):
        group_command = self.group_command
        self.group_feedback = self.hexapod.get_next_feedback(reuse_fbk=self.group_feedback)
        group_feedback = self.group_feedback
        hexapod = self.hexapod

        #getting from current position to starting position
        firstpos = group_feedback.position
        positions = np.zeros((18, 3), dtype=np.float64)
        
        # start in locomotion stance
        positions[:, 0] = firstpos
        positions[:, 1] = firstpos
        positions[:, 2] = np.array(
            [0, 0, -1.57, 0, 0, 1.57, 0, 0, -1.57, 0, 0, 1.57, 0, 0, -1.57, 0, 0, 1.57])

        time_vector = [0, 1, 2]
        trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
        duration = trajectory.duration
        start = time()
        t = time() - start
        while t < duration:
            # Serves to rate limit the loop without calling sleep
            hexapod.get_next_feedback(reuse_fbk=group_feedback)
            t = time() - start
            pos, vel, acc = trajectory.get_state(t)
            #print(pos)
            group_command.position = pos
            hexapod.send_command(group_command)
        
        sleep(1.0)
        
        # move leg 3 and 6
        positions[:, 0] = np.array(
            [0, 0, -1.57, 0, 0, 1.57, 0, 0, -1.57, 0, 0, 1.57, 0, 0, -1.57, 0, 0, 1.57])
        positions[:, 1] = np.array(
            [0, 0, -1.57, 0, 0, 1.57, -0.5, -0.5, -1.57, 0, 0, 1.57, 0, 0, -1.57, 0.18, 0.3, 1.57])
        positions[:, 2] = np.array(
            [0, 0, -1.57, 0, 0, 1.57, -0.91, 0.3 ,-1.25, 0, 0, 1.57, 0, 0, -1.57, 0.33,-0.21,1.33])

        time_vector = [0, 1, 2]
        trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
        duration = trajectory.duration
        start = time()
        t = time() - start
        while t < duration:
            # Serves to rate limit the loop without calling sleep
            #hexapod.get_next_feedback(reuse_fbk=group_feedback)
            t = time() - start
            pos, vel, acc = trajectory.get_state(t)
            #print(pos)
            group_command.position = pos
            hexapod.send_command(group_command)
        
        # move leg 4 and 5
        positions[:, 0] = np.array(
            [0, 0, -1.57, 0, 0, 1.57, -0.91, 0.3 , -1.25, 0, 0, 1.57, 0, 0, -1.57, 0.33,-0.21,1.33])
        positions[:, 1] = np.array(
            [0, 0, -1.57, 0, 0, 1.57, -0.91, 0.3 , -1.25, 0.3, 0.3, 1.57, -0.18, -0.3, -1.57, 0.33,-0.21,1.33])
        positions[:, 2] = np.array(
            [0, 0, -1.57, 0, 0, 1.57, -0.91, 0.3 , -1.25, 0.83, -0.3, 1.28, -0.26, 0.26, -1.19, 0.33,-0.21,1.33])

        time_vector = [0, 1, 2]
        trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
        duration = trajectory.duration
        start = time()
        t = time() - start
        while t < duration:
            # Serves to rate limit the loop without calling sleep
            #hexapod.get_next_feedback(reuse_fbk=group_feedback)
            t = time() - start
            pos, vel, acc = trajectory.get_state(t)
            #print(pos)
            group_command.position = pos
            hexapod.send_command(group_command)
        
        # move leg 1 and 2
        positions[:, 0] = np.array(
            [0, 0, -1.57, 0, 0, 1.57, -0.91, 0.3 , -1.25, 0.83, -0.3, 1.28, -0.26, 0.26, -1.19, 0.33, -0.21, 1.33])
        positions[:, 1] = np.array(
            [0, -0.3, -1.57, 0, 0.3, 1.57, -0.91, 0.3 , -1.25, 0.83, -0.3, 1.28, -0.26, 0.26, -1.19, 0.33, -0.21, 1.33])
        positions[:, 2] = np.array(
            [0., 0.78, -2.44, 0., -0.78, 2.41, -0.91, 0.3 , -1.25, 0.83, -0.3, 1.28, -0.26, 0.26, -1.19, 0.33, -0.21, 1.33])

        time_vector = [0, 1, 2]
        trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
        duration = trajectory.duration
        start = time()
        t = time() - start
        while t < duration:
            # Serves to rate limit the loop without calling sleep
            #hexapod.get_next_feedback(reuse_fbk=group_feedback)
            t = time() - start
            pos, vel, acc = trajectory.get_state(t)
            #print(pos)
            group_command.position = pos
            hexapod.send_command(group_command)'''
    
''' def manipulation2locomotion(self):
        group_command = self.group_command
        self.group_feedback = self.hexapod.get_next_feedback(reuse_fbk=self.group_feedback)
        group_feedback = self.group_feedback
        hexapod = self.hexapod
        #set up the positions array
        positions = np.zeros((18, 3), dtype=np.float64)

        # move leg 1 and 2
        positions[:, 0] = np.array(
            [0., 0.78, -2.44, 0., -0.78, 2.41, -0.91, 0.3 , -1.25, 0.83, -0.3, 1.28, -0.26, 0.26, -1.19, 0.33, -0.21, 1.33])
        positions[:, 1] = np.array(
            [0, -0.2, -1.57, 0, 0.2, 1.57, -0.91, 0.3 , -1.25, 0.83, -0.3, 1.28, -0.26, 0.26, -1.19, 0.33, -0.21, 1.33])
        positions[:, 2] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, -0.91, 0.3 , -1.25, 0.83, -0.3, 1.28, -0.26, 0.26, -1.19, 0.33, -0.21, 1.33])

        time_vector = [0, 1, 2]
        trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
        duration = trajectory.duration
        start = time()
        t = time() - start
        while t < duration:
            # Serves to rate limit the loop without calling sleep
            hexapod.get_next_feedback(reuse_fbk=group_feedback)
            t = time() - start
            pos, vel, acc = trajectory.get_state(t)
            #print(pos)
            group_command.position = pos
            hexapod.send_command(group_command)
        
        # move leg 4 and 3
        positions[:, 0] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, -0.91, 0.3 , -1.25, 0.83, -0.3, 1.28,    -0.26,     0.26,   -1.19,  0.33, -0.21,1.33])
        positions[:, 1] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57,-0.5, -0.5, -1.57, 0.3,   0.3, 1.57,    -0.26,     0.26,   -1.19,  0.33, -0.21,1.33])
        positions[:, 2] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, 0, 0, -1.57,   0,     0, 1.57,    -0.26,     0.26,   -1.19,  0.33, -0.21, 1.33])

        time_vector = [0, 1, 2]
        trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
        duration = trajectory.duration
        start = time()
        t = time() - start
        while t < duration:
            # Serves to rate limit the loop without calling sleep
            hexapod.get_next_feedback(reuse_fbk=group_feedback)
            t = time() - start
            pos, vel, acc = trajectory.get_state(t)
            #print(pos)
            group_command.position = pos
            hexapod.send_command(group_command)
        
        # move leg 5
        positions[:, 0] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, 0, 0, -1.57,   0,     0, 1.57,    -0.26,     0.26,   -1.19,  0.33, -0.21, 1.33])
        positions[:, 1] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, 0, 0, -1.57,   0,     0, 1.57,    -0.18, -0.3, -1.57,  0.33, -0.21, 1.33])
        positions[:, 2] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, 0, 0, -1.57,   0,     0, 1.57,     0, 0, -1.57,  0.33, -0.21, 1.33])

        time_vector = [0, 1, 2]
        trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
        duration = trajectory.duration
        start = time()
        t = time() - start
        while t < duration:
            # Serves to rate limit the loop without calling sleep
            hexapod.get_next_feedback(reuse_fbk=group_feedback)
            t = time() - start
            pos, vel, acc = trajectory.get_state(t)
            #print(pos)
            group_command.position = pos
            hexapod.send_command(group_command)
        
        #move leg 6
        positions[:, 0] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, 0, 0, -1.57,   0,     0, 1.57,     0, 0, -1.57,  0.33, -0.21, 1.33])
        positions[:, 1] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, 0, 0, -1.57,   0,     0, 1.57,     0, 0, -1.57,  0.18, 0.3, 1.57])
        positions[:, 2] = np.array(
            [0, 0.2, -1.57, 0, -0.2, 1.57, 0, 0, -1.57,   0,     0, 1.57,     0, 0, -1.57,  0, 0, 1.57])

        time_vector = [0, 1, 2]
        trajectory = hebi.trajectory.create_trajectory(time_vector, positions)
        duration = trajectory.duration
        start = time()
        t = time() - start
        while t < duration:
            # Serves to rate limit the loop without calling sleep
            hexapod.get_next_feedback(reuse_fbk=group_feedback)
            t = time() - start
            pos, vel, acc = trajectory.get_state(t)
            #print(pos)
            group_command.position = pos
            hexapod.send_command(group_command)'''