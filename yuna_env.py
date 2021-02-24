import pybullet as p
import pybullet_data
import time
import math
import numpy as np
from CPG.calculate_limitcycle import *
from CPG.updateCPGStance import *
from CPG.CPG_controller import *


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# set visualization
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load world

p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF (plane)
planeId = p.loadURDF("plane.urdf")
p.setGravity(0,0,-10)

# load robot model

YunaStartPos = [0,0,0.5]
YunaStartOrientation = p.getQuaternionFromEuler([0,0,0])
Yuna = p.loadURDF("urdf/yuna.urdf",YunaStartPos, YunaStartOrientation)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.saveWorld("yuna_quickworld")

joint_num = p.getNumJoints(Yuna)
actuators = [i for i in range(joint_num) if p.getJointInfo(Yuna,i)[2] != p.JOINT_FIXED]

# p.setRealTimeSimulation(1)
legs = [0.] * len(actuators)
rest_position = np.array([0, 0, -1.57,
                     0, 0, -1.57,
                     0, 0, -1.57,
                     0, 0, 1.57,
                     0, 0, 1.57,
                     0, 0, 1.57])


from setup.xMonsterKinematics import *
xmk = HexapodKinematics()

# CPG
T = 5000 #Time in seconds code operates for
nIter = int(round(T/0.01))
pi = math.pi
#creating cpg dict
cpg = {
    'initLength': 0,
    's': 0.15 * np.ones(6),   # stride length
    'nomX': np.array([0.51589,  0.51589,  0.0575,   0.0575, - 0.45839, - 0.45839]),
    'nomY': np.array([0.23145, - 0.23145,   0.5125, - 0.5125,   0.33105, - 0.33105]),
    'h': 0.2249,
    's1OffsetY': np.array([0.2375*math.sin(pi/6), -0.2375*math.sin(pi/6), 0.1875, -0.1875, 0.2375*math.sin(pi/6), -0.2375*math.sin(pi/6)]),#robot measurement;  distance on y axis from robot center to base_actuator
    's1OffsetAngY': np.array([-pi/3, pi/3, 0, 0, pi/3, -pi/3]),
    'n': 2,  #limit cycle shape 2:standard, 4:super
    'b': np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), #np.array([.4, .4, .4, .4, .4, .4]), #TUNEABLE: step height in radians %1.0
    'scaling': 10, #TUNEABLE: shifts the units into a reasonable range for cpg processing (to avoid numerical issues)
    'shouldersCorr': np.array([-1, 1, -1, 1, -1, 1]),
    'phase_lags': np.array([pi, pi, 0, pi, 0, pi]),
    'dynOffset': np.zeros([3,6]), #Offset on joints developed through constraining
    'dynOffsetInc': np.zeros([3,6]), #Increment since last iteration
    'x': np.zeros([nIter,6]), #TUNEABLE: Initial CPG x-positions
    'y': np.zeros([nIter,6]), #TUNEABLE: Initial CPG y-positions
    'x0': np.zeros([1,6]), #limit cycle center x
    'y0': np.zeros([6,6]), #limit cycle center y
    'legs': np.zeros([1,18]), #Joint angle values
    'elbowsLast': np.zeros([1,6]), #elbow values
    'torques': np.zeros([1,18]), #Joint torque values
    'torqueOffsets': np.zeros([1,18]), #Joint torque offset values
    'gravCompTorques': np.zeros([1,18]), #Joint torque values
    'forces': np.zeros([3,6]), #ee force values
    'gravCompForces': np.zeros([3,6]), #Joint torque values
    'forceStance': np.zeros([1,6]), #grounded legs determined by force
    'CPGStance': np.array([False,False,False,False,False,False]), #grounded legs determined by position (lower tripod)
    'CPGStanceDelta': np.zeros([1,6]), #grounded legs determined by position (lower tripod)
    'CPGStanceBiased': np.zeros([1,6]), #grounded legs determined by position (lower tripod)
    'comm_alpha': 1.0, #commanded alpha in the complementary filter (1-this) is the measured joint angles
    'move': True, #true: walks according to cpg.direction, false: stands in place (will continue to stabilize); leave to true for CPG convergence
    'xmk': xmk, #Snake Monster Kinematics object
    'pose': np.eye(3), #%SO(3) describing ground frame w.r.t world frame
    'R': SE3(np.eye(3),[0, 0, 0]), #SE(3) describing body correction in the ground frame
    'G': np.eye(3), #SO(3) describing ground frame w.r.t world frame
    'tp': np.zeros([4,1]),
    'dynY': 0,
    'vY': 0,
    'direction' : 'forward',
    'fullStepLength' : 20000,
    't' : 0
}


cx = np.array([-1/6, -1/6, 0, 0, 0, 0]) * pi
cy = np.array([0, 0, 0, 0, 0, 0]) * pi


cpg['eePos'] = np.vstack((cpg['nomX'],cpg['nomY'], -cpg['h'] * np.ones([1,6]) )) # R: Compute the EE positions in body frame
ang = cpg['xmk'].getLegIK(cpg['eePos']) #R: This gives the angles corresponding to each of the joints
cpg['nomOffset'] = np.reshape(ang[0:18], [6, 3]).T
cpg['nomOffset'] = cpg['nomOffset'] * cpg['scaling']

# the distance between foot ground trajectory (TUNEABLE)
cpg['foothold_offsetY'] = np.array([0.33145, - 0.33145,   0.5125, - 0.5125,   0.33105, - 0.33105])

# the distance between foot ground trajectory to base actuator
dist = cpg['foothold_offsetY'] - cpg['s1OffsetY']

# calculate the a respective to cx to make stride length of 6 legs to be equal
a = calculate_a(cpg, cx, dist)

cpg['a'] = a * cpg['scaling']
cpg['b'] = cpg['b'] * cpg['scaling']
cpg['cx'] = cx * cpg['scaling']
cpg['cy'] = cy * cpg['scaling']

cpg['K'] = np.array( [[0, -1, -1,  1,  1, -1],
                     [-1,  0,  1, -1, -1,  1],
                     [-1,  1,  0, -1, -1,  1],
                     [ 1, -1, -1,  0,  1, -1],
                     [ 1, -1, -1,  1,  0, -1],
                     [-1,  1,  1, -1, -1,  0]])

# Initialize the x and y values of the cpg cycle
cpg['x'][0, :] = (a * np.array([1, -1, -1, 1, 1, -1]) + cx) * cpg['scaling']
cpg['y'][0, :] = np.zeros(6)
dt = 0.01  # CPG frequency

frame = 0
while 1:
    p.stepSimulation()


    cpg = updateCPGStance(cpg, cpg['t'])
    cpg, positions = CPG(cpg, cpg['t'], dt)

    cpg['t'] += 1

    if cpg['t'] > (cpg['initLength'] + 300):
        cpg['move'] = True
        cpg_position = cpg['legs'][0, :]
        # leg.num correction due to urdf file
        commanded_position[0:3] = cpg_position[0:3]     # 1 to 1
        commanded_position[3:6] = cpg_position[6:9]     # 2 to 3
        commanded_position[6:9] = cpg_position[12:15]   # 3 to 5
        commanded_position[9:12] = cpg_position[3:6]    # 4 to 2
        commanded_position[12:15] = cpg_position[9:12]   # 5 to 4
        commanded_position[15:18] = cpg_position[15:18]   # 6 to 6

    else:
        commanded_position = rest_position

    print(commanded_position)
    forces = [40.] * len(actuators)
    p.setJointMotorControlArray(Yuna, actuators, controlMode=p.POSITION_CONTROL, targetPositions=commanded_position,
                                positionGains=[0.5]*len(actuators),velocityGains=[1]*len(actuators),forces=forces)


    # time.sleep(0.01)
    time.sleep(1. / 240.)
    distance = 2
    yaw = 40
    YunaPos, YunaOrn = p.getBasePositionAndOrientation(Yuna)
    # p.resetDebugVisualizerCamera(distance, yaw, -20, YunaPos)
    frame += 1
# YunaPos, YunaOrn = p.getBasePositionAndOrientation(Yuna)
# print(YunaPos,YunaOrn)
# time.sleep(10)
p.disconnect()

"""
# get motors list
joint_num = p.getNumJoints(Yuna)
print("joint_number:",joint_num)
actuators = []
for j in range(joint_num):
    info = p.getJointInfo(Yuna,j)
    if info[2] == p.JOINT_REVOLUTE:
        actuators.append(j)
        print(info[1])
print(actuators)   # [12, 13, 14, 17, 18, 19, 22, 23, 24, 27, 28, 29, 32, 33, 34, 37, 38, 39]   in  1,3,5,2,4,6 order
"""




