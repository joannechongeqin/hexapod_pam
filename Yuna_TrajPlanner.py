import numpy as np
import hebi
from functions import solveIK, rotz, transxy
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt

# default robot leg end-effecter position w.r.t body frame
eePos = np.array([[0.51589,    0.51589,   0.0575,     0.0575,     -0.45839,   -0.45839],
                  [0.23145,   -0.23145,   0.5125,     -0.5125,    0.33105,    -0.33105],
                  [-0.2249,   -0.2249,    -0.2249,    -0.2249,    -0.2249,    -0.2249]])
class TrajPlanner:
    def __init__(self, neutralPos=eePos):
        self.dutyfactor = 0.5 # percent of the total cycle which a given foot is on the ground = time foot is on the ground (stance phase) / total time of gait cycle
        self.period = 1.0 #  total duration of the gait period 
        self.dt = 1/240 # pybullet default  # time step duration
        self.eePos = neutralPos
        self.eeAng = np.array([0., 0., 0., 0., 0., 0.,]) # the deviation of each leg from neutral position, use 0. to initiate a float type array
        self.array_dim = int(np.around(self.period / self.dt)) # timesteps of a complete period
        self.stance_dim = int(np.around(self.period * self.dutyfactor / self.dt)) # 120
        self.swing_dim = self.array_dim - self.stance_dim # 120
        self.traj_dim = np.maximum(self.swing_dim, self.stance_dim)
        self.clearance = 0.1 # maximum foot clearance when the robot lifts its leg
        self.tripod1 = [0, 3, 4] # leg index for leg 1, 4, 5 (left front, left back, right middle)
        self.tripod2 = [1, 2, 5] # leg index for leg 2, 3, 6 (right front, right back, left middle)
        self.quad1 = [0, 3] 
        self.quad2 = [1, 4]
        self.quad3 = [2, 5]
        self.num_legs = 6
        
    def get_loco_traj(self, init_pose, step_len, course, rotation, flag, timestep):
        '''
        Compute the leg trajectories of all six legs within a stride
        :param init_pose: initial pose of all six legs' task coordinate frame
        :param step_len: step length, translational displacement of the leg's task coordinate frame with respect to the body frame
        :param course: course angle in radians
        :param rotation: rotation angle, rotational displacement of the leg's task coordinate frame with respect to the body frame
        :param flag: use the parity of the flag to determine which set of tripod to move forward, odd flag for tripod1, even flag for tripod2
        :param timestep: current timestep
        :return traj: position of all six legs at a given timestep
        :return end_pose: desired end pose of this trajectory of all six legs' task coordinate frame
        '''
        traj = np.zeros((3, 6)) # initializes a 3x6 matrix to store the computed positions of all six legs at the given timestep
        end_pose = self._get_end_pose(step_len, course, rotation, flag) # a 4x6 matrix, calculate the desired end pose for each leg based on the step length, course, rotation, and the flag
        curve_type = ['swing', 'stance', 'stance', 'swing', 'swing', 'stance'] if flag % 2 == 0 else ['stance', 'swing', 'swing', 'stance', 'stance' , 'swing']

        # print("init_pose:", init_pose)
        # print('end_pose:', end_pose)
        for leg_index in range(6):
            # traj[:,leg_index] gets column leg_index fo the traj matrix
            traj[:,leg_index] = self._compute_traj(init_pose[:,leg_index], end_pose[:,leg_index], curve_type[leg_index], leg_index, timestep)
        # print("traj:", traj)
        return traj, end_pose
    
    def _get_end_pose(self, step_len, course, rotation, flag=0): # use flag to determine which set of tripod to move forward
        '''
        Compute the end pose of each leg's task coordinate frame
        :param step_len: step length, translational displacement of the leg's task coordinate frame with respect to the body frame
        :param course: course angle in radians
        :param rotation: rotation angle, rotational displacement of the leg's task coordinate frame with respect to the body frame
        :param flag: use the parity of the flag to determine which set of tripod to move forward, odd flag for tripod1, even flag for tripod2
        :return end_pose: end pose of all six legs' task coordinate frame
        '''
        end_pos = np.zeros((3, 6)) # initialize a 3x6 array to store the final pos for each of the 6 legs
        end_ang = np.zeros((6,)) # initialize a 1x6 array to store the final orientation for each of the 6 legs
        neutral_pose = np.zeros((4,1)) # a 4x1 array, reference point for calculating the end poses of the legs
        if flag % 2 == 0: # swing tripod1 forward
            for leg_index in self.tripod1:
                # moves tripod1 forward relative to the body
                end_pos[:, leg_index] = np.reshape(transxy(neutral_pose[:3], +step_len/2, course), (3,)) # compute the end position by translate +step_len/2 from the neutral_pose by in the direction specified by course
                end_ang[leg_index] = neutral_pose[3] + rotation / 2 # compute the end orientation
            for leg_index in self.tripod2:
                # moves tripod2 backward relative to the body
                end_pos[:, leg_index] = np.reshape(transxy(neutral_pose[:3], -step_len/2, course), (3,))
                end_ang[leg_index] = neutral_pose[3] - rotation / 2
        else: # swing tripod2 forward
            for leg_index in self.tripod1:
                end_pos[:, leg_index] = np.reshape(transxy(neutral_pose[:3], -step_len/2, course), (3,))
                end_ang[leg_index] = neutral_pose[3] - rotation / 2
            for leg_index in self.tripod2:
                end_pos[:, leg_index] = np.reshape(transxy(neutral_pose[:3], +step_len/2, course), (3,))
                end_ang[leg_index] = neutral_pose[3] + rotation / 2
        
        end_pose = np.vstack((end_pos, end_ang))
        return end_pose

    def pose2pos(self, pose, leg_index): 
        '''
        Compute the position of each leg from the pose of leg's task coordinate frame
        :param pose: 4x1 array, pose of leg's task coordinate frame with respect to body frame
        :param leg_index: int, index of leg
        :return pos: position of leg (xyz)
        '''
        pos = rotz(pos=self.eePos[:,leg_index]+pose[:3], angle=pose[3], pivot=pose[:3])
        return pos

    def _compute_traj(self, init_pose, end_pose, curve_type, leg_index, timestep): # pose of single leg
        '''
        Compute the foot trajectory for a single leg within a stride, and return the position of the foot at a given timestep
        The trajectory planner uses a cycloid curve to generate the foot trajectory, you can use plot_traj.py to visualise this curve
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
        where s stands for step length and c stands for foot clearance

        :param init_pose: initial pose of the each leg's task coordinate with respect to the body frame
        :param end_pose: end pose of the each leg's task coordinate with respect to the body frame
        :param curve_type: 'swing' or 'stance'
        :param leg_index: index of the leg
        :param timestep: correponding timestep of the desired waypoint within a stride trajectory
        :return: position of the foot with respect to the body frame at the given timestep
        '''
        traj = np.zeros((self.swing_dim, 3))
        deltaPose = end_pose - init_pose
        
        if curve_type == 'swing':
            dt = 2 * np.pi / self.swing_dim
            clearance = np.array((0, 0, self.clearance)) # foot clearance
            t = timestep * dt
            pose = init_pose + deltaPose * (t - 2 * np.sin(t)) / (2 * np.pi) # handle translational and angular displacement at the same time
            traj = self.pose2pos(pose, leg_index) + clearance * (1 - 2 * np.cos(t) + 1) / 4
        elif curve_type == 'stance':
            dPose = deltaPose / self.traj_dim
            pose = timestep * dPose + init_pose 
            traj = self.pose2pos(pose, leg_index)
        else:
            raise ValueError('Wrong curve type, the available types are: \'swing\' or \'stance\'.')
        
        return traj
    
    def pam_traj(self, legs_waypoints, total_time):
        n = len(legs_waypoints)
        t = np.linspace(0, total_time, n)
        m = int(total_time / self.dt) + 1
        t_new = np.linspace(0, total_time, m)
        
        trajectory = np.zeros((m, 3, 6), dtype=np.float32)
        
        for leg in range(6):
            for dim in range(3):
                y = np.array([legs_waypoints[i][dim, leg] for i in range(n)])
                dydx = np.gradient(y, t)
                spline = CubicHermiteSpline(t, y, dydx)
                trajectory[:, dim, leg] = spline(t_new)
        
        return trajectory
    
    def pam_tripod_keyframe(self, body_w_init, body_w_final, leg_w_init, leg_w_final, raise_height=0.1):
        # split into 4 keyframes where body moves 1/4 forward each time
        quarter_body_trans = (body_w_final - body_w_init) / 4
        body_keyframe_w = [body_w_init]
        for i in range(4): 
            body_keyframe_w.append(body_keyframe_w[-1] + quarter_body_trans)

        leg_keyframe_w = [leg_w_init]
        
        mid = (leg_w_init + leg_w_final) / 2
        # 1. raise first tripod set
        set1 = leg_w_init.copy()
        for i in self.tripod1: # raise tripod 1 legs
            set1[:, i] = mid[:, i] 
            set1[2, i] = max(set1[2, i], leg_w_final[2, i]) + raise_height
        leg_keyframe_w.append(set1)
        
        # 2. lower first tripod set
        set2 =  set1.copy()
        for i in self.tripod1: # lower tripod 1 legs
            set2[:, i] = leg_w_final[:, i]
        leg_keyframe_w.append(set2)

        # 3. raise second tripod set
        set3 = set2.copy()
        for i in self.tripod2: # raise tripod 2 legs
            set3[:, i] = mid[:, i]
            set3[2, i] = max(set3[2, i], leg_w_final[2, i]) + raise_height
        leg_keyframe_w.append(set3)

        # 4. lower second tripod set
        leg_keyframe_w.append(leg_w_final)
        
        # print("body_keyframe_w:\n", body_keyframe_w)
        # print("leg_keyframe_w:\n", leg_keyframe_w)
        return body_keyframe_w, leg_keyframe_w        

    def pam_press_button_keyframe(self, body_w_init, body_w_final, leg_w_init, leg_w_final, press_leg_idx, raise_height=0.1):
        # assuming can only press with front two legs (index 0 or 1)
        
        # old version: quadrangular gait:
        #   1. move leg2 and leg5 (and trans half of body)
        #   2. move leg3 and leg4 (and trans half of body)
        #   3. move last support leg
        #   4. final leg to press

        # new version:
        #   1. move first set of legs (trans 1/3 of body)
        #   2. move second set of legs (trans 1/3 of body)
        #   3. final leg to press (trans 1/3 of body)

        if press_leg_idx == 1:
            pair1 = [0, 3, 4]
            pair2 = [2, 5]
        elif press_leg_idx == 0:
            pair1 = [1, 2, 5]
            pair2 = [3, 4]

        # for step 1 and 2: split into 4 keyframes where body moves 1/6 forward each time
        quarter_body_trans = (body_w_final - body_w_init) / 6
        body_keyframe_w = [body_w_init]
        for i in range(4): 
            body_keyframe_w.append(body_keyframe_w[-1] + quarter_body_trans)

        leg_keyframe_w = [leg_w_init]
        
        mid = (leg_w_init + leg_w_final) / 2
        # 1. raise first pair
        set1 = leg_w_init.copy()
        for i in pair1: # raise pair 1 legs
            set1[:, i] = mid[:, i] 
            set1[2, i] = max(set1[2, i], leg_w_final[2, i]) + raise_height
        leg_keyframe_w.append(set1)
        
        # 2. lower first tripod set
        set2 =  set1.copy()
        for i in pair1: # lower pair 1 legs
            set2[:, i] = leg_w_final[:, i]
        leg_keyframe_w.append(set2)

        # 3. raise second tripod set
        set3 = set2.copy()
        for i in pair2: # raise pair 2 legs
            set3[:, i] = mid[:, i]
            set3[2, i] = max(set2[2, i], leg_w_final[2, i]) + raise_height
        leg_keyframe_w.append(set3)

        # 4. lower second pair set
        set4 = set3.copy()
        for i in pair2: # lower pair 1 legs
            set4[:, i] = leg_w_final[:, i]
        leg_keyframe_w.append(set4)
        
        # 5. raise last support leg
        # set5 = set4.copy()
        # set5[:, last_support_leg] = mid[:, last_support_leg]
        # set5[2, last_support_leg] = max(set4[2, last_support_leg], leg_w_final[2, last_support_leg]) + raise_height
        # leg_keyframe_w.append(set5)
        # body_keyframe_w.append(body_keyframe_w[-1])

        # # 6. lower last support leg
        # set6 = set5.copy()
        # set6[:, last_support_leg] = leg_w_final[:, last_support_leg]
        # leg_keyframe_w.append(set6)
        # body_keyframe_w.append(body_keyframe_w[-1]) 

        # 7. slowly raise last leg to press
        set7a = set4.copy()
        set7a[2, press_leg_idx] = set4[2, press_leg_idx] + raise_height
        leg_keyframe_w.append(set7a)
        body_keyframe_w.append(body_keyframe_w[-1] + quarter_body_trans)

        set7b = set7a.copy()
        set7b[:, press_leg_idx] = (set7b[:, press_leg_idx] + leg_w_final[:, press_leg_idx]) / 2
        leg_keyframe_w.append(set7b)
        body_keyframe_w.append(body_keyframe_w[-1] + quarter_body_trans)

        # 8. press with last leg
        set8 = set7b.copy()
        set8[:, press_leg_idx] = leg_w_final[:, press_leg_idx]
        leg_keyframe_w.append(set8)
        body_keyframe_w.append(body_keyframe_w[-1])

        print("press_button body_keyframe_w:\n", body_keyframe_w)
        print("press_button leg_keyframe_w:\n", leg_keyframe_w)
        return body_keyframe_w, leg_keyframe_w 
    
    def pam_wave_keyframe(self, body_w_init, body_w_final, leg_w_init, leg_w_final, raise_height=0.1, support_legs = [True] * 6, leg_sequence = [0, 1, 2, 3, 4, 5]):
        num_support_legs = sum(support_legs)
        body_trans_interval = (body_w_final - body_w_init) / num_support_legs / 2
        # only move body when support legs are moving
        # when raising unsupported legs, body stays at the same position

        # make sure to raise unsupported legs last
        start_raising_not_support_legs = False
        for leg_idx in leg_sequence:
            if not support_legs[leg_idx]:
                start_raising_not_support_legs = True
            if start_raising_not_support_legs and support_legs[leg_idx]:
                raise ValueError("Legs not providing support should be raised last")

        body_keyframe_w = [body_w_init]
        leg_keyframe_w = [leg_w_init]
        mid = (leg_w_init + leg_w_final) / 2
        curr = leg_w_init.copy() # current

        def _append_body_keyframe(leg_idx):
            if support_legs[leg_idx]:
                body_keyframe_w.append(body_keyframe_w[-1] + body_trans_interval)
            else:
                body_keyframe_w.append(body_keyframe_w[-1])

        for leg_idx in leg_sequence:
            # raise leg
            curr[:, leg_idx] = mid[:, leg_idx]
            curr[2, leg_idx] = max(curr[2, leg_idx], leg_w_final[2, leg_idx]) + raise_height
            leg_keyframe_w.append(curr.copy())
            _append_body_keyframe(leg_idx)
            # lower leg
            curr[:, leg_idx] = leg_w_final[:, leg_idx]
            leg_keyframe_w.append(curr.copy())
            _append_body_keyframe(leg_idx)

        # print("body_keyframe_w:\n", body_keyframe_w)
        # print("leg_keyframe_w:\n", leg_keyframe_w)
        return body_keyframe_w, leg_keyframe_w   

    def visualize_trajectory(trajectory):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for leg in range(6):
            ax.plot(trajectory[:, 0, leg], trajectory[:, 1, leg], trajectory[:, 2, leg], label=f'Leg {leg+1}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def general_traj(self, waypoints, total_time=1, time_vector=[]):
        '''
        Use Hebi's trajectory interpolation function to generate a series of more dense interpolated trajectories in jointspace
        :param waypoints: a list of waypoints, each waypoint can be either a 3x6 matrix (workspace command) or a 18x1 vector (jointspace command)
        :param total_time: the total time of robot following the trajectory
        :param time_vector: a list of time points, each time point is the time when the robot reaches the corresponding waypoint, therefore the length of time_vector should be the same as the length of waypoints
        :return: a list of interpolated trajectories in jointspace
        '''
        num_pos = len(waypoints) # the number of given positions in a trajectory
        jointspace_command = np.zeros(shape=(18,num_pos))
        # unify all waypoints to joint space
        for i in range(num_pos):
            if np.shape(waypoints[i]) == (3,6): # input workspace command as waypoint
                _, jointspace_command[:,i] = solveIK(waypoints[i]) # discard the first output which is for pybullet environment
            elif np.shape(waypoints[i]) == (18,): # input jointspace command as waypoint
                jointspace_command[:,i] = waypoints[i]
            else:
                raise ValueError('Command that Yuna cannot recognise')
        # print("jointspace_command:", jointspace_command)
        # default time vector, assuming the time is evenly distributed, otherwise please assign customised time vector in argument
        if not time_vector:
            interval = total_time / (num_pos - 1)
            time_vector = [interval * _  for _ in range(num_pos)]
        # create trajectory
        trajectory = hebi.trajectory.create_trajectory(time_vector, jointspace_command)
        duration = trajectory.duration
        len_traj = int(duration / self.dt) + 1
        traj = np.zeros((len_traj,18))
        for i in range(len_traj):
            pos, vel, acc = trajectory.get_state(i*self.dt)
            traj[i] = pos
        return traj

if __name__ == '__main__':
    # plot trajectory of a leg
    import numpy as np
    import matplotlib.pyplot as plt
    # params
    tp = TrajPlanner()
    step_len = 0.2
    course = 0
    rotation = np.pi / 6
    leg_index = 3
    # compute trajectory
    end_pose = tp._get_end_pose(step_len, course, rotation)[:, leg_index]
    init_pose = - end_pose
    traj_swing = np.zeros((tp.swing_dim, 3))
    traj_stance = np.zeros((tp.stance_dim, 3))
    for timestep in range(tp.traj_dim):
        traj_swing[timestep] = tp._compute_traj(init_pose, end_pose, curve_type='swing', leg_index=leg_index, timestep=timestep)
        traj_stance[timestep] = tp._compute_traj(end_pose, init_pose, curve_type='stance', leg_index=leg_index, timestep=timestep)
    traj = np.vstack((traj_swing, traj_stance))
    # plot
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z, s=1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_aspect('equal', 'box')
    plt.show()
    pass