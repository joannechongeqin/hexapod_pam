import math
import numpy as np
import pybullet as p
import pybullet_data
from setup.yunaKinematics import HexapodKinematics
import time
from scipy.spatial import ConvexHull
from matplotlib import path
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


def set_up():
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF (plane)
    planeId = p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def hebi_to_pybullet(hebi_position):
    pybullet_position = np.zeros(18)
    pybullet_position[0:3] = hebi_position[0:3]  # 1 to 1
    pybullet_position[3:6] = hebi_position[6:9]  # 2 to 3
    pybullet_position[6:9] = hebi_position[12:15]  # 3 to 5
    pybullet_position[9:12] = hebi_position[3:6]  # 4 to 2
    pybullet_position[12:15] = hebi_position[9:12]  # 5 to 4
    pybullet_position[15:18] = hebi_position[15:18]  # 6 to 6
    return pybullet_position


def Render_plot(plot_points):
    YunaPos, YunaOrn = p.getBasePositionAndOrientation(Yuna)
    pose = p.getMatrixFromQuaternion(YunaOrn)
    pose = np.asarray(pose)
    pose_matrix = np.reshape(pose, (3, 3))

    p1 = np.dot(pose_matrix, plot_points[:, 0].T) + YunaPos
    p4 = np.dot(pose_matrix, plot_points[:, 1].T) + YunaPos
    p5 = np.dot(pose_matrix, plot_points[:, 2].T) + YunaPos
    com = np.dot(pose_matrix, plot_points[:, 3].T) + YunaPos

    p.addUserDebugLine(p1, p4, lineColorRGB=[0, 0, 1], lineWidth=6.0, lifeTime=120 / 120)
    p.addUserDebugLine(p4, p5, lineColorRGB=[0, 0, 1], lineWidth=6.0, lifeTime=120 / 120)
    p.addUserDebugLine(p5, p1, lineColorRGB=[0, 0, 1], lineWidth=6.0, lifeTime=120 / 120)
    p.addUserDebugPoints([com], pointColorsRGB=[[1, 0, 0]], pointSize=10, lifeTime=120 / 120)

    p.stepSimulation()


def plot_check(deltax, deltay, joint_angles, grounded_legs):
    xmk = HexapodKinematics()

    frames = xmk.getFrames_output(joint_angles.reshape(1, 18))

    c_list = []
    h_list = []
    f_list = []
    l_max = 0.65  # leg1+leg2
    r_2d_min = 0.05
    for leg in range(6):
        b = frames[leg, 0, 0:3, 3]  # the base joint [x, y, z] in the robot frame
        f = frames[leg, 5, 0:3, 3]  # the foot [x, y, z] in the robot frame
        f_list.append(f)
        c = f - b
        l = np.linalg.norm(c)
        c_list.append(c)
        h = frames[leg, 0, 2, 3] - c[2]
        h_list.append(h)
    # Display the workspace for the grounded legs (three legs per time)
    grounded_points = []
    positions = xmk.getLegPositions(joint_angles)
    # if cpg['CPGStanceDelta'][5] == True:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for leg in range(6):
        if grounded_legs[leg] == True:
            # Small circle
            r_2d_min = 0.05
            circle = Circle(xy=(c_list[leg][0], c_list[leg][1]), radius=r_2d_min, alpha=0.1, color='orange')
            ax.add_patch(circle)
            # Big circle
            r_2d_max = math.sqrt(l_max ** 2 - h_list[leg] ** 2)
            circle = Circle(xy=(c_list[leg][0], c_list[leg][1]), radius=r_2d_max, alpha=0.1,
                            color='g' if leg in [0, 3, 4] else 'yellow')
            ax.add_patch(circle)
            # Gravity center
            circle = Circle(xy=(0, 0), radius=0.02, alpha=0.1, color='k')
            ax.add_patch(circle)
            # Support polygon
            grounded_points.append(positions[0:2, leg])
    grounded_points = np.asarray(grounded_points)
    hull = ConvexHull(grounded_points)
    for simplex in hull.simplices:
        plt.plot(grounded_points[simplex, 0], grounded_points[simplex, 1], 'k', lw=0.8)

    circle = Circle(xy=(deltax, deltay), radius=0.02, alpha=0.1, color='r')
    ax.add_patch(circle)
    plt.plot(grounded_points[hull.vertices, 0], grounded_points[hull.vertices, 1], 'k', lw=0.8)
    plt.axis('equal')
    plt.show()


class IK_controller:
    def __init__(self, step_h, step_l, T, ini_eePos):
        self.T = T
        self.step_h = step_h
        self.step_l = step_l
        self.theta = np.pi * np.ones(6)  # theta xz, b

        self.dt = 0
        self.a = self.step_h / self.step_l ** 2
        self.b = -2 * self.a * self.step_l
        self.c = np.zeros(6)
        self.phi = 4 * math.pi / self.T
        self.A = self.step_l / (np.sin(self.phi * self.T / 2) - self.phi * self.T / 2)
        self.phase = [False, False, False, False, False, False]
        self.ini_eePos = ini_eePos
        self.currentPos = self.ini_eePos

    def set_step(self, current_Pos: np.array, next_step: np.array, next_com: np.array):
        """

        :param current_Pos: (3*6) current position of each end effectors respect to the body frame
        :param next_step: (3*6) 基于绝对坐标系下的腿的移动位置
        :param next_com: (1*3) 基于绝对坐标系机器人重心的移动位置
        :return:
        """
        self.ini_eePos = current_Pos
        self.phase = [True for i in range(6)]
        for i in range(6):
            if (next_step[:, i] == [0, 0, 0]).all():
                self.phase[i] = False
        self.dt = 0
        next_com[2] = 0
        gait_step = next_step - next_com.reshape(3, 1)  # 获得在 body frame 下三条空中腿的移动向量
        gait_length = np.sqrt(gait_step[0] ** 2 + gait_step[1] ** 2) + 1.e-5  # in case of length=0 计算步长
        gait_theta = list()
        for i in range(6):
            if self.phase[i]:
                gait_theta.append(np.arctan2(gait_step[1][i], gait_step[0][i]) + np.pi)  # 计算移动的向量与x轴正半轴的角度
            else:
                gait_theta.append(np.arctan2(gait_step[1][i], gait_step[0][i]))

        self.step_l = gait_length / 2
        self.theta = gait_theta

        self.a = self.step_h / self.step_l ** 2

        self.b = -2 * self.a * self.step_l
        self.A = self.step_l / (np.sin(self.phi * self.T / 2) - self.phi * self.T / 2)

    def step_once(self):
        dis = -self.A * (self.phi * self.dt - np.sin(self.phi * self.dt))

        x = -dis * np.cos(self.theta)
        y = -dis * np.sin(self.theta)
        z = -(self.a * dis ** 2 + self.b * dis + self.c)

        xMovement = x
        yMovement = y
        zMovement = z

        swingArray = np.vstack((xMovement, yMovement, zMovement))
        moveArray = np.vstack((-xMovement, -yMovement, np.zeros(6)))
        swingPos = self.ini_eePos + swingArray
        movePos = self.ini_eePos + moveArray

        currentPos = movePos

        for i in range(6):
            if self.phase[i]:
                currentPos[:, i] += -movePos[:, i] + swingPos[:, i]

        currentPos = np.transpose(currentPos)
        self.currentPos = np.array(currentPos)
        self.dt += 1

        if len(self.currentPos) == 6:
            self.currentPos = self.currentPos.T

        return self.currentPos

    def step(self):
        pos_list = list()
        while self.dt < (self.T + 1):
            pos_list.append(self.step_once())
        return pos_list

    def walk(self, current_Pos: np.array):
        """

        :param current_Pos: (3*6) current position of each end effectors respect to the body frame
        :return: pos_list: end effectors 移动轨迹
        """
        pos_list = list()
        next_foothold = list()
        next_com = np.array([0.15, 0.0, -0.2249])
        next_step = np.array([[0.30, 0, 0, 0.30, 0.30, 0],
                              [-0., -0., -0., 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])
        self.set_step(current_Pos, next_step, next_com)
        next_foothold.append(self.next_foothold(current_Pos, next_step, next_com))
        pos_list += self.step()
        next_step = np.array([[0, 0.30, 0.30, 0, 0, 0.30],
                              [-0., -0., -0., 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]])
        current_Pos = pos_list[-1]
        self.set_step(current_Pos, next_step, next_com)
        next_foothold.append(self.next_foothold(current_Pos, next_step, next_com))
        pos_list += self.step()

        return pos_list, next_foothold

    def next_foothold(self, current_Pos: np.array, next_step: np.array, next_com: np.array):
        next_foothold = list()
        for i in range(len(self.phase)):
            if self.phase[i]:
                next_foothold.append(current_Pos[:, i] + next_step[:, i])
        next_foothold.append(next_com.T)
        next_foothold = np.array(next_foothold).T
        return next_foothold


class IK_checker:
    def __init__(self, confidence):
        self.confidence = confidence
        self.leg_max = 0.655 * self.confidence
        self.base_frame = np.array([[0.19467366, 0.10881532, 0.0242],
                                    [0.19157366, -0.11418468, 0.0242],
                                    [0.0031, 0.17300001, 0.0242],
                                    [-0.0031, -0.17300001, 0.0242],
                                    [-0.19157366, 0.11418468, 0.0242],
                                    [-0.19467366, -0.10881532, 0.0242]]).T

    def check(self, current_Pos: np.array, next_step: np.array, next_com: np.array, grounded_legs: list[bool]):
        """

        :param current_Pos:  (3*6) current position of each end effectors respect to the body frame
        :param next_step: (3*6) 基于绝对坐标系下的腿的移动位置
        :param next_com: (1*3) 基于绝对坐标系机器人重心的移动位置
        :param grounded_legs: (1*6) 立于地面的脚
        :return:
        """
        # body move
        leg_max = self.leg_max * self.confidence
        c_frame = current_Pos - self.base_frame
        polygon = list()
        for leg in range(len(grounded_legs)):
            if grounded_legs[leg]:
                if (next_com[0] - c_frame[0][leg]) ** 2 + (next_com[1] - c_frame[1][leg]) ** 2 + (
                        next_com[2] - c_frame[2][leg]) ** 2 > leg_max ** 2:
                    return False
                polygon.append(current_Pos[0:2, leg])
        hull = ConvexHull(polygon)
        polygon = [polygon[i] for i in hull.vertices]
        polygon = path.Path(polygon)
        in_or_out = polygon.contains_point((next_com[0], next_com[1]))
        if not in_or_out:
            return False
        # foot move
        next_c_frame = current_Pos + next_step - next_com.reshape(3, 1) - self.base_frame
        for i in range(len(grounded_legs)):
            if not grounded_legs[i]:
                if (0 - next_c_frame[0, i]) ** 2 + (0 - next_c_frame[1, i]) ** 2 + (
                        0 - next_c_frame[2, i]) ** 2 >= leg_max ** 2:
                    return False
        return True

    # def check(self, next_step_world: np.array, next_com_world: np.array, joint_angles: np.array,
    #           grounded_legs: list[bool]):
    #     """
    #
    #     :param next_step_world: (3*6) 世界坐标系下腿的位移向量
    #     :param next_com_world: (1*3) 世界坐标系下身体的位移向量
    #     :param joint_angles: (1*18) 各个关节的角度
    #     :param grounded_legs: (1*6) 目前落地的脚
    #     :return:
    #     """
    #     # body move
    #     frames = self.xmk.getHexapodFrames(joint_angles.reshape(1, 18))
    #     leg_max = self.leg_max * self.confidence
    #     base_frame = frames[0][:, :3, 3]
    #     end_effectors_frame = frames[3][:, :3, 3]
    #     c_frame = end_effectors_frame - base_frame
    #     polygon = list()
    #     for leg in range(len(grounded_legs)):
    #         if grounded_legs[leg]:
    #             if (next_com_world[0] - c_frame[leg][0]) ** 2 + (next_com_world[1] - c_frame[leg][1]) ** 2 + (
    #                     next_com_world[2] - c_frame[leg][2]) ** 2 > leg_max ** 2:
    #                 return False
    #             polygon.append(end_effectors_frame[leg][0:2])
    #     hull = ConvexHull(polygon)
    #     polygon = [polygon[i] for i in hull.vertices]
    #     polygon = path.Path(polygon)
    #     in_or_out = polygon.contains_point((next_com_world[0], next_com_world[1]))
    #     if not in_or_out:
    #         return False
    #     # foot move
    #     next_step_world = next_step_world.T
    #     end_effectors_frame = end_effectors_frame + next_step_world
    #     end_effectors_frame = (end_effectors_frame.T - next_com_world.reshape((3, 1))).T
    #     for i in range(len(grounded_legs)):
    #         if not grounded_legs[i]:
    #             c = end_effectors_frame[:][i] - base_frame[:][i]
    #             if (0 - c[0]) ** 2 + (0 - c[1]) ** 2 + (0 - c[2]) ** 2 >= leg_max ** 2:
    #                 return False
    #     return True


class IK:
    def __init__(self):
        self.T = 50
        self.step_h = 0.15 * np.ones(6)
        self.step_l = 0.12 * np.ones(6)
        self.confidence = 1
        self.h = 0.2249
        self.ini_eePos = np.array([[0.51589, 0.51589, 0.0575, 0.0575, - 0.45839, - 0.45839],
                                   [0.23145, - 0.23145, 0.5125, - 0.5125, 0.33105, - 0.33105],
                                   [-self.h, -self.h, -self.h, -self.h, -self.h, -self.h]])
        self.current_pos = self.ini_eePos
        self.ikController = IK_controller(self.step_h, self.step_l, self.T, self.ini_eePos)
        self.ikChecker = IK_checker(self.confidence)

    def world_to_body(self, position):
        # TODO: 将 world 坐标系下的向量转换为 body 坐标系下
        return position

    def move_step(self, current_Pos, next_step, next_com, grounded_legs):
        if self.ikChecker.check(current_Pos, next_step, next_com, grounded_legs):
            # current_Pos = self.world_to_body(current_Pos_world)
            # next_step = self.world_to_body(next_step_world)
            # next_com = self.world_to_body(next_com_world)
            # phase = [not leg for leg in grounded_legs]
            self.ikController.set_step(current_Pos, next_step, next_com)
            pos_list = self.ikController.step()
            self.current_pos = pos_list[-1]
            return pos_list
        return False


if __name__ == '__main__':
    set_up()
    YunaStartPos = [0, 0, 0.5]  # initial position of robot - CHANGABLE
    YunaStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
    Yuna = p.loadURDF("../urdf/yuna.urdf", YunaStartPos, YunaStartOrientation)

    joint_num = p.getNumJoints(Yuna)
    actuators = [i for i in range(joint_num) if p.getJointInfo(Yuna, i)[2] != p.JOINT_FIXED]
    forces = [60.] * len(actuators)
    legs = [0.] * len(actuators)
    xmk = HexapodKinematics()
    ik = IK()
    frame = 0
    current_ang = xmk.getLegIK(ik.ini_eePos)
    while True:
        p.stepSimulation()
        YunaPos, YunaOrn = p.getBasePositionAndOrientation(Yuna)
        if frame > 200:
            next_step = np.array([[0.30, 0, 0, 0.30, 0.30, 0],
                                  [-0., -0., -0., 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]])
            ground_leg = [False, True, True, False, False, True]
            ik.ikController.phase = [not i for i in ground_leg]
            next_foothold = ik.ikController.next_foothold(ik.current_pos, next_step, np.array([0.15, 0.0, -0.2249]))
            pos_list = ik.move_step(ik.current_pos, next_step, np.array([0.15, 0.0, -0.2249]), ground_leg)
            Render_plot(next_foothold)
            for i in pos_list:
                current_ang = xmk.getLegIK(i)
                commanded_position = hebi_to_pybullet(current_ang)
                p.setJointMotorControlArray(Yuna, actuators, controlMode=p.POSITION_CONTROL,
                                            targetPositions=commanded_position,
                                            positionGains=[0.5] * len(actuators), velocityGains=[1] * len(actuators),
                                            forces=forces)
                p.stepSimulation()
                time.sleep(1 / 240)
            next_step = np.array([[0, 0.30, 0.30, 0, 0, 0.30],
                                  [-0., -0., -0., 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]])
            ground_leg = [not i for i in ground_leg]
            ik.ikController.phase = [not i for i in ground_leg]
            next_foothold = ik.ikController.next_foothold(ik.current_pos, next_step, np.array([0.15, 0.0, -0.2249]))
            pos_list = ik.move_step(ik.current_pos, next_step, np.array([0.15, 0.0, -0.2249]), ground_leg)
            Render_plot(next_foothold)
            for i in pos_list:
                current_ang = xmk.getLegIK(i)
                commanded_position = hebi_to_pybullet(current_ang)
                p.setJointMotorControlArray(Yuna, actuators, controlMode=p.POSITION_CONTROL,
                                            targetPositions=commanded_position,
                                            positionGains=[0.5] * len(actuators), velocityGains=[1] * len(actuators),
                                            forces=forces)
                p.stepSimulation()
                time.sleep(1 / 240)
        commanded_position = hebi_to_pybullet(current_ang)
        p.setJointMotorControlArray(Yuna, actuators, controlMode=p.POSITION_CONTROL, targetPositions=commanded_position,
                                    positionGains=[0.5] * len(actuators), velocityGains=[1] * len(actuators),
                                    forces=forces)
        frame += 1

    p.disconnect()
