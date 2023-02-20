import numpy as np
from robot_setup.yunaKinematics import HexapodKinematics
import hebi

class TrajPlanner:
    def __init__(self):
        self.dutyfactor = 0.5
        self.period = 1.0 
        self.dt = 1/240 # pybullet default
        self.array_dim = int(np.around(self.period / self.dt))#timesteps of a complete period
        self.support_dim = int(np.around(self.period * self.dutyfactor / self.dt)) #120
        self.swing_dim = self.array_dim - self.support_dim #120
        self.clearance = 0.1 # maximum foot clearance when the robot lifts its leg
        self.xmk = HexapodKinematics()

    def general_traj(self, waypoints, total_time=1, time_vector=[]): 
        # input: a series of trajectories, output: a series of more dense interpolated trajectories
        num_pos = np.shape(waypoints)[0]#the number of given positions in a trajectory
        jointspace_command = np.zeros(shape=(18,num_pos))
        for i in range(num_pos):
            _, jointspace_command[:,i] = self._solveIK(waypoints[i]) # discard the first output which is for pybullet environment
        if not time_vector:
            interval = total_time / (num_pos - 1)
            time_vector = [interval * _  for _ in range(num_pos)] #assuming the time is evenly distributed, otherwise please assign customised time vector in argument
        trajectory = hebi.trajectory.create_trajectory(time_vector, jointspace_command)
        duration = trajectory.duration
        len_traj = int(duration / self.dt) + 1
        traj = np.zeros((len_traj,3,6))
        for i in range(len_traj):
            pos, vel, acc = trajectory.get_state(i*self.dt)
            traj[i] = self._solveFK(pos)
        return traj        

    def walk_swing_traj(self, init_pos, end_pos):
        via_pos = (init_pos + end_pos) / 2
        via_pos[2] += self.clearance
        t = np.ones((7, self.swing_dim))
        tf = self.swing_dim * self.dt # finish time, usually 0.5s
        time = np.linspace(0, tf, self.swing_dim)
        for i in range(7):
            t[i,:] = np.power(time, i)
        a_0 = init_pos
        a_1 = np.zeros(3)
        a_2 = np.zeros(3)
        a_3 = (2 / (tf ** 3)) * (32 * (via_pos - init_pos) - 11 * (end_pos - init_pos))
        a_4 = -(3 / (tf ** 4)) * (64 * (via_pos - init_pos) - 27 * (end_pos - init_pos))
        a_5 = (3 / (tf ** 5)) * (64 * (via_pos - init_pos) - 30 * (end_pos - init_pos))
        a_6 = -(32 / (tf ** 6)) * (2 * (via_pos - init_pos) - (end_pos - init_pos))
        traj = np.stack([a_0, a_1, a_2, a_3, a_4, a_5, a_6], axis=-1).dot(t)
        return np.transpose(traj)

    def walk_support_traj(self, init_pos, end_pos):
        traj = np.linspace(init_pos, end_pos, self.support_dim, axis=1)
        return np.transpose(traj)

    def turn_swing_traj(self, neutral_pos, init_ang, end_ang):
        traj = np.zeros((self.swing_dim, 3))
        init_pos = self._rot(neutral_pos, init_ang)
        end_pos = self._rot(neutral_pos, end_ang)
        via_pos = self._rot(neutral_pos, (init_ang + end_ang) / 2)
        via_pos[2] += self.clearance
        t = np.ones((7, self.swing_dim))
        tf = self.swing_dim * self.dt # finish time, usually 0.5s
        time = np.linspace(0, tf, self.swing_dim)
        for i in range(7):
            t[i,:] = np.power(time, i)
        a_0 = init_pos
        a_1 = np.zeros(3)
        a_2 = np.zeros(3)
        a_3 = (2 / (tf ** 3)) * (32 * (via_pos - init_pos) - 11 * (end_pos - init_pos))
        a_4 = -(3 / (tf ** 4)) * (64 * (via_pos - init_pos) - 27 * (end_pos - init_pos))
        a_5 = (3 / (tf ** 5)) * (64 * (via_pos - init_pos) - 30 * (end_pos - init_pos))
        a_6 = -(32 / (tf ** 6)) * (2 * (via_pos - init_pos) - (end_pos - init_pos))
        swing_traj = np.transpose(np.stack([a_0, a_1, a_2, a_3, a_4, a_5, a_6], axis=-1).dot(t))
        support_traj = self.turn_support_traj(neutral_pos, init_ang, end_ang)
        traj[:, :2], traj[:, 2] = support_traj[:,:2], swing_traj[:,2]
        return traj

    def turn_support_traj(self, neutral_pos, init_ang, end_ang):
        traj = np.zeros((3, self.support_dim))
        angles = np.linspace(init_ang, end_ang, self.support_dim)
        for i in range(self.support_dim):
            traj[:,i] = self._rot(neutral_pos, angles[i])
        return np.transpose(traj)

    def _rot(self, pos, angle):
        c, s = np.cos(angle), np.sin(angle)
        rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pos_ = np.matmul(rot_z, pos)
        return pos_
    
    def _solveIK(self, workspace_command):
        jointspace_command2hebi = self.xmk.getLegIK(workspace_command)
        jointspace_command2bullet = jointspace_command2hebi[[0,1,2,6,7,8,12,13,14,3,4,5,9,10,11,15,16,17,]].copy()# reshaped the IK result: 123456->135246
        return jointspace_command2bullet, jointspace_command2hebi

    def _solveFK(self, jointspace_command2hebi):
        workspace_command = self.xmk.getLegPositions(np.array([jointspace_command2hebi]))
        return workspace_command






