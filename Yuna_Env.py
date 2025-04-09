import pybullet as p
import pybullet_data
import numpy as np
import os, sys
import time
import robot_setup
from robot_setup.yunaKinematics import *
from functions import hebi2bullet, bullet2hebi, solveIK, solveFK
import matplotlib.pyplot as plt
from zed_camera import ZedCamera

np.set_printoptions(precision=4, suppress=True)

h = 0.12
eePos = np.array(  [[0.51589,    0.51589,   0.0575,     0.0575,     -0.45839,   -0.45839],
                    [0.23145,   -0.23145,   0.5125,     -0.5125,    0.33105,    -0.33105],
                    [   -h,         -h,         -h,         -h,         -h,         -h]])

class Map:
    def __init__(self, map_range=5.0, map_resolution=0.05, pybullet_on=False):
        self.map_range = map_range
        self.map_resolution = map_resolution
        self.pybullet_on = pybullet_on
        self.height_map = self._generate_height_map()
        self.eef_height_offset = 0.025 # offset 0.025m from ground (when at initial position, all eef pos are around 0.025m above ground)

    def _generate_height_map(self):
        x_coords = np.arange(-self.map_range, self.map_range, self.map_resolution)
        y_coords = np.arange(-self.map_range, self.map_range, self.map_resolution)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        x_grid = np.round(x_grid, 3)
        y_grid = np.round(y_grid, 3)

        height_map = np.zeros_like(x_grid)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                x, y = x_grid[i, j], y_grid[i, j]
                ray_start = [x, y, 10]  # ray starts at 10m above ground
                ray_end = [x, y, -10]   # ray ends at 10m below ground

                if self.pybullet_on:
                    ray_result = p.rayTest(ray_start, ray_end)
                    if ray_result[0][0] != -1 and ray_result[0][3][2] > 1e-6:  # if ray hits something
                            height_map[i, j] = ray_result[0][3][2]  # z-coordinate of the hit point
                            # print(f"Ray hit at ({x}, {y}) idx ({i}, {j}) with height {height_map[i, j]}")
                else:
                    height_map[i, j] = 0.0

        return height_map
    
    def get_height_at(self, x, y):
        # note currently rounding up, maybe can round to nearest cell?
        y_idx = int((x + self.map_range) / self.map_resolution)
        x_idx = int((y + self.map_range) / self.map_resolution)
        # print(f"getting height at ({x}, {y}) idx ({x_idx}, {y_idx}) = {self.height_map[x_idx, y_idx]}")
        if not (0 <= x_idx < self.height_map.shape[0] and 0 <= y_idx < self.height_map.shape[1]):
            print(f"Coordinates ({x}, {y}) are out of bounds for the height map.")
        return self.height_map[x_idx, y_idx] + self.eef_height_offset
    
    def get_heights_at(self, arr_of_xy):
        return np.array([self.get_height_at(x, y) for x, y in arr_of_xy])
    
    def get_max_height_below_base(self, center_x, center_y):
        # estimated body size: 390mm in y direction, 600mm in x direction
        base_len_x, base_len_y = 0.6, 0.4
        x_min = center_x - base_len_x / 2
        x_max = center_x + base_len_x / 2
        y_min = center_y - base_len_y / 2
        y_max = center_y + base_len_y / 2
        x_points = np.arange(x_min, x_max + self.map_resolution, self.map_resolution)
        y_points = np.arange(y_min, y_max + self.map_resolution, self.map_resolution)
        grid_points = np.array([[x, y] for y in y_points for x in x_points])
        heights_below_body_area = self.get_heights_at(grid_points)
        # print("x_points: ", x_points)
        # print("y_points: ", y_points)
        # print("grid_points_arr: ", grid_points_arr)
        # print("heights_below_body_area: ", heights_below_body_area)
        return np.max(heights_below_body_area)
    
    # TODO: get variance --> such that free legs xy dont optimize so close to edges
    # def get_variance_at(self, center_x, center_y, window_size=5):
    #     half_window = window_size // 2
    #     x_min = center_x - half_window * self.map_resolution
    #     x_max = center_x + half_window * self.map_resolution
    #     y_min = center_y - half_window * self.map_resolution
    #     y_max = center_y + half_window * self.map_resolution

    #     # Clip points to be within the map range
    #     x_points = np.clip(np.arange(x_min, x_max + self.map_resolution, self.map_resolution), 
    #                        -self.map_range, self.map_range)
    #     y_points = np.clip(np.arange(y_min, y_max + self.map_resolution, self.map_resolution), 
    #                        -self.map_range, self.map_range)

    #     # Generate all grid points in the window
    #     grid_points = np.array([[x, y] for y in y_points for x in x_points])

    #     # Retrieve heights and compute the variance
    #     heights = self.get_heights_at(grid_points)
    #     # print("heights: ", heights)
    #     return np.round(np.var(heights), 5)
    
    # def  get_variances_at(self, arr_of_xy, window_size=5):
    #     print(f"finding variances at {arr_of_xy}")
    #     return np.array([self.get_variance_at(x, y, window_size) for x, y in arr_of_xy])
    
    def plot(self):
        plt.figure(figsize=(10, 7))
        plt.imshow(self.height_map, extent=(-self.map_range, self.map_range, -self.map_range, self.map_range), origin='lower', cmap='viridis')
        plt.colorbar(label='Height')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Height Map")
        plt.grid(True)
        plt.show()
    
    def save_map(self, file_path: str):
        data = {
            "map_range": self.map_range,
            "map_resolution": self.map_resolution,
            "height_map": self.height_map
        }
        np.savez(file_path, **data)
        print(f"Map data saved to {file_path}.npz")

    def load_map(self, file_path: str):
        data = np.load(file_path + ".npz")
        
        self.map_range = float(data["map_range"])
        self.map_resolution = float(data["map_resolution"])
        self.height_map = data["height_map"]

        print(f"Map data loaded from {file_path}.npz")

class YunaEnv:
    def __init__(self, real_robot_control=True, zed_on=False, pybullet_on=True, visualiser=True, camerafollow=False, 
                    eePos=eePos, goal=[], 
                    load_fyp_map=True, map_range=2.5, map_resolution=0.05):
        self.real_robot_control = real_robot_control
        self.visualiser = visualiser
        self.camerafollow = camerafollow
        self.dt = 1/240
        self.xmk, self.imu, self.hexapod, self.fbk_imu, self.fbk_hp, self.group_command, self.group_feedback = self.robot_connect()
        self.error = np.zeros((18,))
        # self.all_reaction_forces = []
        # self.all_joint_torques = []
        self.complex_terrain = False
        self.pybullet_on = True # pybullet_on # must be set to True if not will kill some functions below
        
        self.eePos = eePos.copy() # initial eePos wrt robot frame

        self.load_fyp_map = load_fyp_map
        self.goal = goal
        self.map_range = map_range
        self.map_resolution = map_resolution

        if self.pybullet_on:
            self._load_env()
        self._init_robot() # will have data of self.body_pos_w, self.body_orn_w
        
        self.zed_on = zed_on
        if self.zed_on:
            time.sleep(2)
            self.zed = ZedCamera()

    def step(self, targetPositions, iteration=1, initializing=False, sleep='auto'):
        '''
        Advance the simulation and physical robot by one step
        :param targetPositions: the target position of the robot, either in the form of workspace command (shape=(3,6)) or jointspace command (shape=(18,))
        :param iteration: the number of iterations to reach the target position, default is 1. If the given target position is not possible to reach within 1 step, more than 1 iteration could be set
        :param sleep: the time to sleep after each step, default is 'auto', which means the time to sleep is automatically calculated based on the given frequency
        return: None
        '''
        # judge the input target position is workspace command or jointspace command
        if np.shape(targetPositions) == (3,6): # workspace command
            jointspace_command2bullet, jointspace_command2hebi = solveIK(targetPositions)
        elif np.shape(targetPositions) == (18,): # jointspace command
            jointspace_command2bullet, jointspace_command2hebi = hebi2bullet(targetPositions), targetPositions
        else:
            raise ValueError('Command that Yuna cannot recognise, please input either workspace command whose shape(targetPositions)=(3,6), or jointspace command whose shape(targetPositions)=(18,)')
        
        for i in range(iteration): # iteration is usually 1, but if the given target position is not possible to reach within 1 step, more than 1 iteration could be set
            t_start = time.perf_counter()
            # real robot control
            if self.real_robot_control:
                self.group_command.position = jointspace_command2hebi - 0.3 * self.error
                self.hexapod.send_command(self.group_command)
                while True:
                    try:
                        self.group_feedback = self.hexapod.get_next_feedback(reuse_fbk=self.group_feedback)
                        self.error = self.group_feedback.position - jointspace_command2hebi
                        break
                    except:
                        pass

            # pybullet control
            if self.pybullet_on:
                p.setJointMotorControlArray(
                    bodyIndex=self.YunaID, 
                    jointIndices=self.actuator, 
                    controlMode=p.POSITION_CONTROL, 
                    targetPositions=jointspace_command2bullet,
                    # forces=self.forces
                    # forces = [60]*18
                    )
                p.stepSimulation()

                if not initializing and self.zed_on and self.real_robot_control:
                    # print("self.body_pos_w, self.body_orn_w: ", self.body_pos_w, self.body_orn_w)
                    # print("frame_from_zed: ", self.zed.get_robot_frame_wrt_world())
                    self.body_pos_w, self.body_orn_w = self.zed.get_robot_frame_wrt_world()
                    p.resetBasePositionAndOrientation(self.YunaID, self.body_pos_w, self.body_orn_w)
                else:
                    self.body_pos_w, self.body_orn_w = p.getBasePositionAndOrientation(self.YunaID)

                self.body_orn_w = np.array(self.body_orn_w)

                if self.camerafollow:
                    self._cam_follow()
            t_stop = time.perf_counter()
            t_step = t_stop - t_start
            if sleep == 'auto':
                time.sleep(max(0, self.dt - t_step))
            else:
                time.sleep(sleep)

    def close(self):
        '''
        Close the pybullet simulation, disable the real robot motors, and terminate the program
        :return: None
        '''
        if self.real_robot_control:
                arr = np.zeros([1, 18])[0]
                self.group_command.effort = arr
                self.group_command.position = np.nan * arr
                self.group_command.velocity_limit_max = arr
                self.group_command.velocity_limit_min = arr
                self.hexapod.send_command(self.group_command)

        if self.zed_on:
            self.zed.close()

        if self.pybullet_on:
            try:
                p.disconnect()
            except p.error as e:
                print('Termination of simulation failed:', e)
        sys.exit()

    def robot_connect(self):
        '''
        Initialise connection to the real robot
        :return: xmk, imu, hexapod, fbk_imu, fbk_hp, group_command, group_feedback
        '''
        if self.real_robot_control:
            xmk, imu, hexapod, fbk_imu, fbk_hp = robot_setup.setup_xmonster()
            group_command = hebi.GroupCommand(hexapod.size)
            group_feedback = hebi.GroupFeedback(hexapod.size)
            hexapod.feedback_frequency = 100.0
            hexapod.command_lifetime = 0
            while True:
                group_feedback = hexapod.get_next_feedback(reuse_fbk=group_feedback)
                if type(group_feedback) != None:
                    break
            return xmk, imu, hexapod, fbk_imu, fbk_hp, group_command, group_feedback
        else:
            return HexapodKinematics(), False, False, False, False, False, False

    def load_rectangular_body(self, position, size, color):
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
        rec_id = p.createMultiBody(baseCollisionShapeIndex=wall_shape, basePosition=position, baseVisualShapeIndex=visual_shape)
        return rec_id

    def _load_env(self):
        '''
        Load and initialise the pybullet simulation environment
        :return: None
        '''
        # initialise interface
        if self.visualiser:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        # physical parameters
        self.gravity = -9.81
        self.friction = 0.8
        # load ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if self.complex_terrain:
            heightPerturbationRange = 0.1
            numHeightfieldRows = 256
            numHeightfieldColumns = 256
            resolution = 0.05
            heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
            for j in range (int(numHeightfieldColumns/2)):
                for i in range (int(numHeightfieldRows/2) ):
                    height = random.uniform(0,heightPerturbationRange)
                    heightfieldData[2*i+2*j*numHeightfieldRows]=height
                    heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
                    heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
                    heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
            terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[resolution,resolution,1], 
                                                  heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, 
                                                  numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
            self.groundID  = p.createMultiBody(0, terrainShape)
        else:
            self.groundID = p.loadURDF('plane.urdf')

        p.setGravity(0, 0, self.gravity)
        p.changeDynamics(self.groundID, -1, lateralFriction=self.friction)

        origin = [0, 0, 0]
        axis_length = 0.15
        lineWidth = 8
        p.addUserDebugLine(origin, [axis_length, 0, 0], [1, 0, 0], lineWidth=lineWidth)  # X-axis (Red)
        p.addUserDebugLine(origin, [0, axis_length, 0], [0, 1, 0], lineWidth=lineWidth)  # Y-axis (Green)
        p.addUserDebugLine(origin, [0, 0, axis_length], [0, 0, 1], lineWidth=lineWidth)  # Z-axis (Blue)
    
        if self.load_fyp_map:
            step_color = [0.4, 0.58, 0.93, 1]
            wall_color = [0.286, 0.435, 0.729, 1]
            button_color = [1, 0, 0, 1]
            self.rec1 = self.load_rectangular_body([1., 0, 0], [0.3, .75, 0.2], step_color)
            self.rec2 = self.load_rectangular_body([1.3, 0, 0], [0.2, .75, 0.3], step_color)
            self.wall = self.load_rectangular_body([1.44, 0, 0], [0.025, .75, 1.5], wall_color)
            self.button = self.load_rectangular_body([1.42, -0.3, 0.65], [0.01, .05, 0.05], button_color)
            p.changeDynamics(self.rec1, -1, lateralFriction=self.friction)
            p.changeDynamics(self.rec2, -1, lateralFriction=self.friction)
            if len(self.goal) > 0:
                p.addUserDebugPoints(pointPositions=self.goal, pointColorsRGB=[[0.5, 0.5, 0.5]]*len(self.goal), pointSize=20, lifeTime=0)
            # check body edges
            # p.addUserDebugPoints(pointPositions=[[0.3, 0.2, 0.15]], pointColorsRGB=[[0, 1, 0]], pointSize=20, lifeTime=0)
            # p.addUserDebugPoints(pointPositions=[[-0.3, 0.2, 0.15]], pointColorsRGB=[[0, 1, 0]], pointSize=20, lifeTime=0)
            # p.addUserDebugPoints(pointPositions=[[0.3, -0.2, 0.15]], pointColorsRGB=[[0, 1, 0]], pointSize=20, lifeTime=0)
            # p.addUserDebugPoints(pointPositions=[[-0.3, -0.2, 0.15]], pointColorsRGB=[[0, 1, 0]], pointSize=20, lifeTime=0)

        self.height_map = Map(map_range=self.map_range, map_resolution=self.map_resolution, pybullet_on=self.pybullet_on) # generate a 2.5D height map for the environment

        # load Yuna robot
        Yuna_init_pos = [0,0,0.5]
        Yuna_init_orn = p.getQuaternionFromEuler([0,0,0])
        Yuna_file_path = os.path.abspath(os.path.dirname(__file__)) + '/urdf/yuna.urdf'
        self.YunaID = p.loadURDF(Yuna_file_path, Yuna_init_pos, Yuna_init_orn)
        self.joint_num = p.getNumJoints(self.YunaID) # 41
        self.actuator = [i for i in range(self.joint_num) if p.getJointInfo(self.YunaID,i)[2] != p.JOINT_FIXED] # 18 DOF
        # print("actuators info:", [p.getJointInfo(self.YunaID, joint)[1] for joint in self.actuator])
        # self.forces = [38 if "shoulder" in p.getJointInfo(self.YunaID, joint)[1].decode('utf-8') else 20 for joint in self.actuator]

        if self.visualiser:
            self._add_reference_line()
            self._cam_follow()
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            
    def load_wall(self, wall_size=[1.5, 0.03, 0.75], wall_position = [0, .6, 0.75], wall_orientation = [0, 0, 0, 1.0]):
        wall_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_size)
        self.wallID = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_shape, basePosition=wall_position, baseOrientation=wall_orientation)
        p.changeDynamics(self.wallID, -1, lateralFriction=self.friction)

    def _init_robot(self):
        '''
        Initialise the robot to the neutral position in the pybullet simulation
        :return: None
        '''
        # parameters
        init_pos = self.eePos.copy()
        self.step(init_pos, initializing=True, iteration=65, sleep='auto')
        self.body_pos_w, self.body_orn_w = p.getBasePositionAndOrientation(self.YunaID) # use pybullet to get initial body position and orientation in world frame
        self.body_pos_w = np.array(self.body_pos_w)
        self.body_orn_w = np.array(self.body_orn_w)
        # for i in self.actuator:
        #     p.enableJointForceTorqueSensor(self.YunaID, i, 1) # enable force/torque sensor
    
    def getMatrixFromQuaternion(self, orn):
        return np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    def get_body_matrix(self):
        '''
        Get homogeneous transformation matrix of body frame wrt world frame
        '''
        rot_mat = self.getMatrixFromQuaternion(self.body_orn_w)
        body_mat = np.eye(4)
        body_mat[:3, :3] = rot_mat
        body_mat[:3, 3] = self.body_pos_w
        return body_mat
    
    def _cam_follow(self):
        '''
        Follow the robot with the camera in the pybullet simulation
        :return: None
        '''
        def _get_body_pose(self):
            '''
            Get the position and orientation of the robot in the pybullet simulation
            :return pos: position of the robot in world frame
            :return orn: orientation of the robot in world frame in Euler angles
            '''
            pos, orn = p.getBasePositionAndOrientation(self.YunaID)
            return pos, p.getEulerFromQuaternion(orn)
        cam_pos, cam_orn = _get_body_pose(self)
        # p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=np.rad2deg(cam_orn[2])-90, cameraPitch=-35, cameraTargetPosition=cam_pos)#
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=-90, cameraPitch=-35, cameraTargetPosition=cam_pos)

    def _add_reference_line(self):
        '''
        Add some reference lines to the pybullet simulation
        :return: None
        '''
        p.addUserDebugLine(lineFromXYZ=[-100,-100,0], lineToXYZ=[100,100,0], lineColorRGB=[0.5,0,0], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[-100,100,0], lineToXYZ=[100,-100,0], lineColorRGB=[0.5,0,0], lineWidth=1)
        # p.addUserDebugLine(lineFromXYZ=[-100,-173.2,0], lineToXYZ=[100,173.2,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        # p.addUserDebugLine(lineFromXYZ=[-100,173.2,0], lineToXYZ=[100,-173.2,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        # p.addUserDebugLine(lineFromXYZ=[-173.2,-100,0], lineToXYZ=[173.2,100,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        # p.addUserDebugLine(lineFromXYZ=[-173.2,100,0], lineToXYZ=[173.2,-100,0], lineColorRGB=[0,0,0.5], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[-100,0,0], lineToXYZ=[100,0,0], lineColorRGB=[0,0,0], lineWidth=1)
        p.addUserDebugLine(lineFromXYZ=[0,-100,0], lineToXYZ=[0,100,0], lineColorRGB=[0,0,0], lineWidth=1)
        
    # def add_y_ref_line_at_height(self, height, lineColorRGB=[1,0,1]):
    #     p.addUserDebugLine(lineFromXYZ=[0,-100,height], lineToXYZ=[0,100,height], lineColorRGB=lineColorRGB, lineWidth=1)
    
    # def add_ref_points_wrt_body_frame(self, points_b, pointSize=5, lifeTime=10):
    #     '''
    #     :param points_b: points wrt body frame, 3x6 (workspace) or 18x1 (jointspace)
    #     '''
    #     if shape(points_b) == (18,):
    #         points_b = solveFK(points_b)
    #     elif shape(points_b) != (3, 6):
    #         raise ValueError('points_b should be either 3x6 or 1x18')
        
    #     WTB = self.get_body_matrix()
    #     points_w = np.dot(WTB, np.vstack((points_b, np.ones((1, 6)))))[0:3, :]
    #     ref_points = [list(points_w[:, i]) for i in range(points_w.shape[1])]
    #     pointColorsRGB = [
    #         [1, 0, 0],   # Red
    #         [0, 1, 0],   # Green
    #         [0, 0, 1],  # Blue
    #         [1, 1, 0],   # Yellow
    #         [0, 1, 1],   # Cyan
    #         [1, 0, 1]   # Magenta
    #     ]
    #     p.addUserDebugPoints(pointPositions=ref_points, pointColorsRGB=pointColorsRGB, pointSize=pointSize, lifeTime=lifeTime)
    
    # def add_body_ref_points_wrt_body_frame(self, points_b, pointSize=5, lifeTime=10):
    #     WTB = self.get_body_matrix()
    #     # print(points_b)
    #     points_w = np.dot(WTB, np.vstack((points_b, np.ones((1, 6)))))[0:3, :]
    #     p.addUserDebugPoints(pointPositions=points_w, pointColorsRGB=[[1, 0, 0]] * len(points_w), pointSize=pointSize, lifeTime=lifeTime)
    
    # def add_body_frame_ref_point(self): # calculated body_frame, might have error when compared to actual one in pybullet
    #     p.addUserDebugPoints(pointPositions=[np.array(self.body_pos_w)], pointColorsRGB=[[0.5, 0.5, 0.5]], pointSize=10, lifeTime=0)
                
    def get_robot_config(self):
        '''
        Get the robot joint configuration
        :return robot_config: robot joint configuration
        '''
        if self.real_robot_control:
            robot_config = self.group_feedback.position
        else:
            robot_config = bullet2hebi(np.array([p.getJointState(self.YunaID, i)[0] for i in self.actuator]))

        return robot_config

    def get_leg_pos(self):
        '''
        Get the robot's leg xyz pos wrt to body frame
        return leg_pos: leg positions 3x6 
        '''
        joint_hebi = self.get_robot_config()
        xmk = HexapodKinematics()
        leg_pos = xmk.getLegPositions(np.array([joint_hebi]))
        return leg_pos
    
    def get_leg_pos_in_world_frame(self):
        '''
        Get the robot's leg xyz pos wrt to world frame
        return leg_pos: leg positions 3x6 
        '''
        leg_pos = self.get_leg_pos()
        WTB = self.get_body_matrix()
        leg_pos_w = np.dot(WTB, np.vstack((leg_pos, np.ones((1, 6)))))[0:3, :]
        return leg_pos_w
    
    def get_elbow_pos(self):
        '''
        Get the robot's elbow xyz pos wrt BODY frame
        return leg_pos: elbow positions 3x6 
        '''
        joint_hebi = self.get_robot_config()
        xmk = HexapodKinematics()
        elbow_pos = xmk.getElbowPositions(np.array([joint_hebi]))
        return elbow_pos
    
if __name__=='__main__':
    # test code
    # yunaenv = YunaEnv(real_robot_control=1, zed_on=True)
    yunaenv = YunaEnv(real_robot_control=0, zed_on=False)  
    print("init_base_pos: ", yunaenv.body_pos_w) # (0, 0, 0.1426)
    
    try:
        while True:
            yunaenv.step(yunaenv.eePos, iteration=1, sleep='auto')
            print("base_pos: ", yunaenv.body_pos_w)

    except KeyboardInterrupt:
        print("Exiting...")

    # height_map = yunaenv.height_map
    # height_map.plot()

    # x = 0
    # for i in range(40):
    #     x = round(x + 0.05, 5)
    #     var_at_edge = height_map.get_variance_at(x, 0)
    #     print(f"Variance at ({x}, 0): {var_at_edge}")

    # height_map.save_map("height_map")

    # new_map_instance = Map()
    # new_map_instance.load_map("height_map")
    # for i in range(10):
    #     x, y = np.random.uniform(-2.5, 2.5), np.random.uniform(-2.5, 2.5)
    #     print(f"Height at ({x}, {y}): {new_map_instance.get_height_at(x, y)}")
    # new_map_instance.plot()

    # print("init_base_pos: ", yunaenv.body_pos_w) # (0, 0, 0.1426)
    # print("init_base_matrix: ", yunaenv.get_body_matrix())
    # print("init_leg_pos_r: ", yunaenv.get_leg_pos())
    # # [[ 0.51637698  0.51636734  0.05752237  0.05750151 -0.45838417 -0.45850756]
    # #  [ 0.2316928  -0.23172229  0.51306432 -0.51305886  0.33115791 -0.33117548]
    # #  [-0.12009861 -0.12009305 -0.120094   -0.12008067 -0.11029625 -0.11044209]]
    # print("init_leg_pos_w: ", yunaenv.get_leg_pos_in_world_frame())
    # # [[ 0.5149191   0.5147523   0.05616759  0.05579873 -0.4598487  -0.46019582]
    # #  [ 0.23102842 -0.23238657  0.51255405 -0.51356891  0.33082626 -0.33150703]
    # #  [ 0.02536965  0.02562286  0.02256538  0.02312707  0.02947116  0.02967863]]

    yunaenv.close()
