import pybullet as p
import pybullet_data
import os
import time
import robot_setup
from robot_setup.yunaKinematics import *

class YunaEnv:
    def __init__(self, visualiser=True, camerafollow=True, real_robot_control=True):
        self.real_robot_control = real_robot_control
        self.visualiser = visualiser
        self.camerafollow = camerafollow
        self.dt = 1 /240
        self.xmk, self.imu, self.hexapod, self.fbk_imu, self.fbk_hp, self.group_command, self.group_feedback = self.robot_connect()
        self._load_env()
        self._init_robot()

    def step(self, targetPositions, iteration=1, sleep='auto'):
        jointspace_command2bullet, jointspace_command2hebi = self._solveIK(targetPositions)
        for i in range(iteration):
            t_start = time.perf_counter()
            p.setJointMotorControlArray(
                bodyIndex=self.YunaID, 
                jointIndices=self.actuator, 
                controlMode=p.POSITION_CONTROL, 
                targetPositions=jointspace_command2bullet)

            if self.real_robot_control:
                self.group_command.position = jointspace_command2hebi
                self.hexapod.send_command(self.group_command)
                self.group_feedback = self.hexapod.get_next_feedback(reuse_fbk=self.group_feedback)

            p.stepSimulation()
            self._cam_follow()
            t_stop = time.perf_counter()
            t_step = t_stop - t_start
            if sleep == 'auto':
                time.sleep(max(0, self.dt - t_step))
            else:
                time.sleep(sleep)

    def close(self):
        try:
            p.disconnect()
        except p.error as e:
            print('Termination of simulation failed:', e)

    def robot_connect(self):
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

    def _load_env(self):
        #initialise interface
        if self.visualiser:
            self.physicsClient = p.connect(p.GUI)
            p.setRealTimeSimulation(1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        #physical parameters
        self.gravity = -9.81
        self.friction = 0.7
        #load ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.groundID = p.loadURDF('plane.urdf')
        p.setGravity(0, 0, self.gravity)
        p.changeDynamics(self.groundID, -1, lateralFriction=self.friction)
        #load Yuna robot
        Yuna_init_pos = [0,0,0.5]
        Yuna_init_orn = p.getQuaternionFromEuler([0,0,0])
        Yuna_file_path = os.path.abspath(os.path.dirname(__file__)) + '/urdf/yuna.urdf'
        self.YunaID = p.loadURDF(Yuna_file_path, Yuna_init_pos, Yuna_init_orn)
        self.joint_num = p.getNumJoints(self.YunaID)
        self.actuator = [i for i in range(self.joint_num) if p.getJointInfo(self.YunaID,i)[2] != p.JOINT_FIXED]
        
        if self.visualiser:
            self._add_reference_line()
            self._cam_follow()
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            
    def _init_robot(self):
        # parameters
        self.h = 0.2249# body height
        self.eePos = np.array( [[0.51589,    0.51589,   0.0575,     0.0575,     -0.45839,   -0.45839],
                                [0.23145,   -0.23145,   0.5125,     -0.5125,    0.33105,    -0.33105],
                                [-self.h,   -self.h,    -self.h,    -self.h,    -self.h,    -self.h]])# neutral position for the robot
        init_pos = self.eePos
        self.step(init_pos, iteration=65, sleep='auto')

    def _solveIK(self, workspace_command):
        jointspace_command2hebi = self.xmk.getLegIK(workspace_command)
        jointspace_command2bullet = jointspace_command2hebi[[0,1,2,6,7,8,12,13,14,3,4,5,9,10,11,15,16,17,]].copy()# reshaped the IK result: 123456->135246
        return jointspace_command2bullet, jointspace_command2hebi
  
    def _cam_follow(self):
        cam_pos, cam_orn = self._get_pose()
        if not self.camerafollow:
            cam_pos = [0, 0, 0.5]
        p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=np.rad2deg(cam_orn[2])-90, cameraPitch=-35, cameraTargetPosition=cam_pos)

    def _get_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.YunaID)
        return pos, p.getEulerFromQuaternion(orn)

    def _add_reference_line(self):
        p.addUserDebugLine(lineFromXYZ=[-100,-100,0], lineToXYZ=[100,100,0], lineColorRGB=[0.5,0,0], lineWidth=10)
        p.addUserDebugLine(lineFromXYZ=[-100,100,0], lineToXYZ=[100,-100,0], lineColorRGB=[0.5,0,0], lineWidth=10)
        p.addUserDebugLine(lineFromXYZ=[-100,-173.2,0], lineToXYZ=[100,173.2,0], lineColorRGB=[0,0,0.5], lineWidth=10)
        p.addUserDebugLine(lineFromXYZ=[-100,173.2,0], lineToXYZ=[100,-173.2,0], lineColorRGB=[0,0,0.5], lineWidth=10)
        p.addUserDebugLine(lineFromXYZ=[-173.2,-100,0], lineToXYZ=[173.2,100,0], lineColorRGB=[0,0,0.5], lineWidth=10)
        p.addUserDebugLine(lineFromXYZ=[-173.2,100,0], lineToXYZ=[173.2,-100,0], lineColorRGB=[0,0,0.5], lineWidth=10)
        p.addUserDebugLine(lineFromXYZ=[-100,0,0], lineToXYZ=[100,0,0], lineColorRGB=[0,0,0], lineWidth=10)
        p.addUserDebugLine(lineFromXYZ=[0,-100,0], lineToXYZ=[0,100,0], lineColorRGB=[0,0,0], lineWidth=10)


if __name__=='__main__':
    #test code
    yunaenv = YunaEnv()
    time.sleep(20)
    yunaenv.close()
