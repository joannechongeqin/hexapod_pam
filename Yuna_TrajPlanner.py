import numpy as np
from robot_setup.yunaKinematics import HexapodKinematics
import hebi
from functions import solveIK, rot

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

    def general_traj(self, waypoints, total_time=1, time_vector=[]): # input: a series of trajectories in workspace or jointspace, output: a series of more dense interpolated trajectories in jointspace
        num_pos = len(waypoints)#the number of given positions in a trajectory
        jointspace_command = np.zeros(shape=(18,num_pos))
        for i in range(num_pos):
            if np.shape(waypoints[i]) == (3,6):# input workspace command as waypoint
                _, jointspace_command[:,i] = solveIK(waypoints[i]) # discard the first output which is for pybullet environment
            elif np.shape(waypoints[i]) == (18,):# input jointspace command as waypoint
                jointspace_command[:,i] = waypoints[i]
            else:
                raise ValueError('Command that Yuna cannot recognise')
        if not time_vector:
            interval = total_time / (num_pos - 1)
            time_vector = [interval * _  for _ in range(num_pos)] #assuming the time is evenly distributed, otherwise please assign customised time vector in argument
        trajectory = hebi.trajectory.create_trajectory(time_vector, jointspace_command)
        duration = trajectory.duration
        len_traj = int(duration / self.dt) + 1
        traj = np.zeros((len_traj,18))
        for i in range(len_traj):
            pos, vel, acc = trajectory.get_state(i*self.dt)
            traj[i] = pos
        return traj   
    '''
    Using a cycloid curve to generate the foot trajectory, you can use plot_traj.py to visualise this curve
    The cycloid parametric equation is:
        x = r * (t - sin(t))
        y = r * (1 - cos(t))
    The height (y_max) of this cycloid is 2*r and length (x_max) is 2*pi*r

    If there is a point rigidly attached to the centre of this circle and the distance is R, the trajectory of this point is:
        x = r * (t - R * sin(t))
        y = r * (1 - R * cos(t))

    Suppose the velocity of circle centre is v0, the angular velocity of the circle is w0, we have:
        v0 = w0 * r
    The points with the same x coordinates with circle centre have the velocity of:
        v = v0 + w0 * (y - yc) = v0 - w0 * R
    where y is the y coodinate of the point, yc is the y coordinate of the circle centre
    If R = 0, v is the velocity of circle centre and equals to v0, 
    if R = r, v is the velocity of the point contacting the ground, which is 0
    Our desired v for the foot trajectory is -v0, and the corresponding R = 2
    
    If we want to rescale it to use as a foot trajectory, then the resized equation is:
        x = s * (t - 2 * sin(t)) / (2 * pi)
        y = c * (1 - 2 * cos(t) + 1) / 4
    where s stands for stride length and c stands for foot clearance
    '''
    def walk_swing_traj(self, init_pos, end_pos): # ellipse trajectory
        stride = end_pos - init_pos # a vector indicating the direction and length
        clearance = np.array((0, 0, self.clearance)) #foot clearance
        dt = 2 * np.pi / self.swing_dim
        traj = np.zeros((3, self.swing_dim))
        for i in range(self.swing_dim):
            t = i * dt
            traj[:, i] = init_pos + stride * (t - 2 * np.sin(t)) / (2 * np.pi) + clearance * (1 - 2 * np.cos(t) + 1) / 4
        return np.transpose(traj)

    def walk_support_traj(self, init_pos, end_pos):
        traj = np.linspace(init_pos, end_pos, self.support_dim, axis=1)
        return np.transpose(traj)

    def turn_swing_traj(self, neutral_pos, init_ang, end_ang):
        stride = end_ang - init_ang
        clearance = np.array((0, 0, self.clearance))
        dt = 2 * np.pi / self.swing_dim
        traj = np.zeros((3, self.support_dim))
        for i in range(self.swing_dim):
            t = i * dt
            ang = init_ang + stride * (t - 2 * np.sin(t)) / (2 * np.pi)
            pos = rot(neutral_pos, ang)
            traj[:, i] = pos + clearance * (1 - 2 * np.cos(t) + 1) / 4
        return np.transpose(traj)

    def turn_support_traj(self, neutral_pos, init_ang, end_ang):
        traj = np.zeros((3, self.support_dim))
        angles = np.linspace(init_ang, end_ang, self.support_dim)
        for i in range(self.support_dim):
            traj[:,i] = rot(neutral_pos, angles[i])
        return np.transpose(traj)