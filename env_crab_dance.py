import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from copy import copy
import time
import numpy as np
#import hebi
import tools
import matplotlib.pyplot as plt
import Functions.TaskCPG as TaskCPG
import Functions.SMIK as SMIK

#Setup Modules and Kinematics Object
from robot_setup.yunaKinematics import *
xmk = HexapodKinematics()

from robot_setup.yunaAnimatronics import *
anm = HexapodAnimatronics()

from os import startfile

# Colors for plots
colors_endForce = ["purple","blue","green","gold","darkorange","red","black"]
colors_jointTorque = ["navy","blue","cornflowerblue","purple","darkviolet","violet","goldenrod","gold","yellow","darkgreen","forestgreen","lime","maroon","red","hotpink","darkorange","orange","sandybrown"]

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

# set visualization
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# load world

# Slope of plane
plane_slope = [0,0,0]
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF (plane)
planeId = p.loadURDF("plane.urdf",baseOrientation=p.getQuaternionFromEuler(plane_slope))
p.setGravity(0,0,-10)

# load robot model

YunaStartPos = [0,0,0.3]
YunaStartOrientation = p.getQuaternionFromEuler([0,0,math.pi])
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

# create stairs
# box11l = 0.29/2
# box11w = 2
# box11h = 0.07
# sh_box11 = p.createCollisionShape(p.GEOM_BOX,halfExtents=[box11l,box11w,box11h])
# stl = box11l * 2
# sth = box11h * 2
# for k in range(10):
#     p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_box11,
#                        basePosition=[-0.5-stl*k, 0+k/200, k*sth], baseOrientation=[0.0, 0.0, 0.0, 1])

cmd = tools.CommandStruct()

T = 60
dt = 0.02 / 10
nIter = round(T / dt)
t_wait = 500

plotF = np.zeros([1+6+1,nIter])
plotT = np.zeros([1+18+1, nIter])
plotM = np.zeros([1+1, nIter])
plotTEq = np.zeros([1+1, nIter])
plotTmax = np.zeros([1+1, nIter])
plotPosition = np.zeros([1+5, nIter])   

commanded_position = np.zeros(18)
a = np.zeros(6)
f = np.zeros(6)
l = np.zeros(6)

x, y, z, alpha, beta, gamma = 0, 0, 0, 0, 0, 0
x_dot, y_dot, z_dot, alpha_dot, beta_dot = 0, 0, 0, 0, 0
x_lim, y_lim, z_lim, alpha_lim, beta_lim = 0.26, 0.05, 0.05, 0.2, 0.05 

P_base = np.zeros([3,6])
    
# Standing on 3 legs (1,4,5)
    
Pfixed = np.array([[ 0.37025384,  0.42114878,  0.02131709, -0.02957786, -0.42945307, -0.37855813],
                    [ 0.40000873, -0.40001041,  0.49999926, -0.50000094,  0.39998939, -0.39999107],
                    [-0.21009839, -0.11753963, -0.10671368, -0.19927244, -0.18844531, -0.09588656]])

P_dance = np.array([[[0.6,  0.7,  0.1, 0.1, -0.3, -0.3],    # Start 0
                    [ 0.35, -0.1,  0.45, -0.45,  0.42, -0.42],
                    [0, 0.2, -0.2, -0.2, -0.2, -0.2]],
    
                    [[0.6,  0.8,  0.1, 0.1, -0.3, -0.3],    # Right 1
                    [ 0.3, -0.2,  0.45, -0.45,  0.42, -0.42],
                    [-0.15, 0.1, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.6,  0.7,  0.1, 0.1, -0.3, -0.3],    # Up 2
                    [ 0.35, -0.2,  0.45, -0.45,  0.42, -0.42],
                    [0.35, 0.45, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.6,  0.7,  0.1, 0.1, -0.3, -0.3],    # Left 3
                    [ 0.22, -0.15,  0.45, -0.45,  0.42, -0.42],
                    [-0.18, 0.1, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.6,  0.7,  0.1, 0.1, -0.3, -0.3],    # Up 4
                    [ 0.27, -0.18,  0.45, -0.45,  0.42, -0.42],
                    [ 0.25, 0.4, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.6,  0.75,  0.1, 0.1, -0.3, -0.3],    # Right 5
                    [ 0.35, -0.2,  0.45, -0.45,  0.42, -0.42],
                    [ 0, 0.1, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.69,  0.65,  0.1, 0.1, -0.3, -0.3],    # Up 6
                    [ 0.1, -0.13,  0.45, -0.45,  0.42, -0.42],
                    [ 0.48, 0.3, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.7,  0.6,  0.1, 0.1, -0.3, -0.3],    # Left 7
                    [ 0.13, -0.13,  0.45, -0.45,  0.42, -0.42],
                    [ 0, 0.27, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.57,  0.52,  0.1, 0.1, -0.3, -0.3],    # Up 8
                    [ 0, -0.5,  0.45, -0.45,  0.42, -0.42],
                    [ 0.14, 0.35, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.7,  0.64,  0.1, 0.1, -0.3, -0.3],    # Right 9
                    [ 0.02, -0.52,  0.45, -0.45,  0.42, -0.42],
                    [ 0.2, 0.1, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.68,  0.64,  0.1, 0.1, -0.3, -0.3],    # Up 10
                    [ 0.03, -0.4,  0.45, -0.45,  0.42, -0.42],
                    [ 0.45, 0.4, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.68,  0.6,  0.1, 0.1, -0.3, -0.3],    # Left 11
                    [ 0.05, -0.38,  0.45, -0.45,  0.42, -0.42],
                    [ 0.1, 0.0, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.57,  0.57,  0.1, 0.1, -0.3, -0.3],    # Up 12
                    [ 0.07, -0.42,  0.45, -0.45,  0.42, -0.42],
                    [ 0.55, 0.4, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.55,  0.6,  0.1, 0.1, -0.3, -0.3],    # Right 13
                    [ 0.15, 0.03,  0.45, -0.45,  0.42, -0.42],
                    [ -0.13, 0.2, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.5,  0.5,  0.1, 0.1, -0.3, -0.3],    # Up 14
                    [ 0.43, -0.15,  0.45, -0.45,  0.42, -0.42],
                    [ 0.1, 0.55, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.65,  0.6,  0.1, 0.1, -0.3, -0.3],    # Left 15
                    [ 0.03, -0.32,  0.45, -0.45,  0.42, -0.42],
                    [ 0.07, 0.18, -0.2, -0.2, -0.2, -0.2]],
                    
                    [[0.6,  0.64,  0.1, 0.1, -0.3, -0.3],    # Up 16
                    [ 0.0, -0.4,  0.45, -0.45,  0.42, -0.42],
                    [ -0.07, 0.08, -0.2, -0.2, -0.2, -0.2]]
                    
                    ])

bodypose_dance = np.array([[0.05, -0.1, 0.05, 0.15, -0.1, 0.2], # Start 0
                  [0.15, -0.02, 0, 0, -0.1, 0.2], # Right 1
                  [0.05, -0.1, 0.2, 0.15, -0.1, 0], # Up 2
                  [-0.1, -0.1, 0.0, 0, -0.1, -0.1], # Left 3
                  [-0.05, -0.1, 0.2, 0.3, 0, 0.1], # Up 4
                  [0.13, -0.1, -0.02, 0, 0.05, 0.23], # Right 5
                  [0.05, -0.1, 0.2, 0.3, -0.05, -0.15], # Up 6
                  [-0.12, -0.1, 0, 0, 0, -0.1], # Left 7
                  [0.08, -0.11, 0.08, 0.5, 0.4, -0.3], # Up 8
                  [0.15, -0.1, -0.05, 0, 0.25, 0], # Right 9
                  [0.05, -0.1, 0.15, 0.3, 0.2, -0.1], # Up 10
                  [-0.12, -0.1, 0, -0.2, 0, -0.3], # Left 11
                  [-0.01, -0.1, 0.1, 0.4, 0, -0.1], # Up 12
                  [0.08, -0.12, -0.02, -0.2, 0, 0.3], # Right 13
                  [0.01, -0.1, 0.11, 0.32, -0.2, 0.1], # Up 14
                  [-0.05, -0.1, -0.05, 0, -0.1, -0.4], # Left 15
                  [0.1, -0.1, 0.1, 0.3, 0.1, -0.25] # Up 16
                #   [-0.12, -0.1, 0, 0, 0, 0] # Right 17
                  ])

pose_count = len(bodypose_dance)

control = 0

grounded_feet = [0, 3, 4]

def rotx(theta):
    return np.array([[1, 0, 0, 0],
                     [0, cos(theta), -sin(theta), 0],
                     [0, sin(theta), cos(theta), 0],
                     [0, 0, 0, 1]])

def roty(theta):
    #Homogeneous transform matrix for a rotation about y
    return np.array([[cos(theta), 0, sin(theta), 0],
                     [0, 1, 0, 0],
                     [-sin(theta), 0, cos(theta), 0],
                     [0, 0, 0, 1]])
    
def rotz(theta):
    return np.array([[cos(theta), -sin(theta), 0, 0],
                     [sin(theta), cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])   

def periodic_body(wt):
    if math.sin(wt) >= 0:
        return abs(math.sin(wt/2))
    else:
        return (1 - math.cos(wt))/2   
    
def periodic_claws(wt):
    if math.sin(wt) >= 0:
        return 1 - abs(math.cos(wt/2))
    if math.sin(wt) < 0:
        return (1 - math.cos(wt))/2        

vx = 0 * np.ones((1, nIter))
vy = 0 * np.ones((1, nIter))
w = 0 * np.ones((1, nIter))
# pitch = 0.027069654743169115 * np.ones((1, nIter))
pitch = 0. * np.ones((1, nIter))

cpg = TaskCPG(dt, nIter, vx, vy, w, pitch)

# startfile("Noisestorm - Crab Rave.mp3")

P_base[:] = P_dance[control]
[x, y, z, alpha, beta, gamma] = bodypose_dance[control]

for t in range(nIter):

    p.stepSimulation()

    # get the yuna's pose feedback
    YunaPos, YunaOrn = p.getBasePositionAndOrientation(Yuna)
    pose = p.getMatrixFromQuaternion(YunaOrn)
    euler = p.getEulerFromQuaternion(YunaOrn)  # roll, pitch, yaw
    # print('euler',euler)
    pose = np.asarray(pose)
    pose_matrix = np.reshape(pose, (3, 3))

    tStart = time.perf_counter()

    correction = 1
    
    # Slow-mo
    # time.sleep(0.05)

    # Generate transformation matrices
    
    # Rotation matrices 4x4
    R_base_body = np.matmul(rotz(gamma), np.matmul(rotx(alpha), roty(beta)))
    
    # Transformation matrices
    # T_base_body 
    T_base_body = np.identity(4)
    T_base_body[0:3, 3] = [x, y, z]
    T_base_body[0:3, 0:3] = R_base_body[0:3, 0:3]
    
    # CPG
    cpg.update(t)

    P = cpg.getEndeffectorPos()
    P = P / 100 # convert cm to m
    # P_base = P[[1, 0, 2], :] # interchange x and y
    
    # Convert P_base to feetpositions_base
    feetpos_base = np.zeros([6,3])
    for i in range(6):
        feetpos_base[i,0] = -P_base[1,i]
        feetpos_base[i,1] = P_base[0,i]
        feetpos_base[i,2] = P_base[2,i]
    
    # Transform feetpositions_base to get feetpositions_body
    feetpos_body = np.zeros([6,3])
    # Grounded Feet
    for i in grounded_feet: 
        feetpos_body[i] = np.matmul(np.linalg.inv(T_base_body), [feetpos_base[i][0], feetpos_base[i][1], feetpos_base[i][2], 1])[0:3] 

    # Feet in the air
    for i in np.setdiff1d(range(6), grounded_feet):
        feetpos_body[i] = np.matmul(np.linalg.inv(T_base_body), [feetpos_base[i][0], feetpos_base[i][1], feetpos_base[i][2], 1])[0:3]     
        
    # Convert feetpositions_body to P_body
    P_body = np.zeros([3,6])
    for i in range(6):
            P_body[0,i] = feetpos_body[i,1]
            P_body[1,i] = -feetpos_body[i,0]
            P_body[2,i] = feetpos_body[i,2]    
    
    # Inverse kinematics of P_body
    ang = xmk.getLegIK(P_body)  # 1 2 3 4 5 6
    
    cpg_position = ang
    
    commanded_position[0:3] = cpg_position[0:3]  # 1 for 1
    commanded_position[3:6] = cpg_position[6:9]  # 2 for 3
    commanded_position[6:9] = cpg_position[12:15]  # 3 for 5
    commanded_position[9:12] = cpg_position[3:6]  # 4 for 2
    commanded_position[12:15] = cpg_position[9:12]  # 5 for 4
    commanded_position[15:18] = cpg_position[15:18]  # 6 for 6

    # Assign Force Limits
    forces = np.zeros(18)
    jointTorqueLims = [60, 60, 60]
    if t > t_wait:
        
        for i in range(len(actuators)):
            # Base Joint
            if i % 3 == 0:
                forces[i] = jointTorqueLims[0]
            # Shoulder Joint
            if i % 3 == 1:
                forces[i] = jointTorqueLims[1]
            # Elbow Joint
            if i % 3 == 2:
                forces[i] = jointTorqueLims[2]    
    else:
        forces = [60.] * len(actuators) # default 60   
        
    # forces = [80.] * len(actuators) # default 60       
                     
    p.setJointMotorControlArray(Yuna, actuators, controlMode=p.POSITION_CONTROL, targetPositions=commanded_position,
                                positionGains=[0.5]*len(actuators),velocityGains=[1]*len(actuators),forces=forces)
    
    # Get Joint Torques   
    torques_un = np.zeros([1,18])
    torques = np.zeros([1,18])
    torques_p = np.zeros([1,18])
    
    k = 0
    for i in actuators:
        torques_un[0,k] = p.getJointState(Yuna, i)[3]
        k += 1

    # Reverse the permutation
    # 1 3 5 2 4 6
    # 1 2 3 4 5 6
    
    torques[0,0:3] = torques_un[0,0:3]  # 1 for 1
    torques[0,3:6] = torques_un[0,9:12]  # 2 for 4
    torques[0,6:9] = torques_un[0,3:6]  # 3 for 2
    torques[0,9:12] = torques_un[0,12:15]  # 4 for 5
    torques[0,12:15] = torques_un[0,6:9]  # 5 for 3
    torques[0,15:18] = torques_un[0,15:18]  # 6 for 6
    
    # Get angles
    angles = np.zeros([1,18])
    angles[0] = ang
    
    # Z-component (Body frame) of force applied by the foot
    l = xmk.getLegTorques(angles, torques)[:,2]
    
    # De-noising (Do not include for real robot)
    a = np.multiply(a, 0.8) + np.multiply(0.2,l)
    
    # No De-noising required for real robot.
    a = l
    
    # Z-component (Body frame) of force applied on the foot (de-noised)
    f = np.multiply(-1, a)
    
    # Indices of feet on the ground
    grounded_feet = xmk.getContactLegs1(a)
    
    if grounded_feet == [0,0,0]:
        correction = 0
        grounded_feet = xmk.getContactLegs(a)
    
    # print(correction)
    
    # Calculate average of f for feet on the ground
    f_avg = 0
    for i in grounded_feet: f_avg += f[i]
    f_avg = f_avg / 3   
    
    torques_p = 0.8 * torques_p + 0.2 * torques

    # Calculate normal vector of plane containing feet
    normal = np.cross((feetpos_body[grounded_feet[2]] - feetpos_body[grounded_feet[0]]), (feetpos_body[grounded_feet[1]] - feetpos_body[grounded_feet[0]]))  # z vector of frame on the ground in body frame
    n_hat = normal / np.linalg.norm(normal)
    
    # Body coordinate frame
    [x_hat, y_hat, z_hat] = np.identity(3)
    
    # Determine pitch (World coordinates) and roll (Body coordinates)
    beta_measure = -np.arcsin(np.dot(n_hat, x_hat))
    alpha_measure = -np.arcsin(np.dot(np.cross(n_hat, x_hat)/cos(beta_measure), z_hat))
    
    # print(x, y, z, alpha, beta)

    # Behaviour of Manipulability vs alpha and beta
    
    # dR/dalpha  
    diff_R_alpha = [[0                               ,  0              ,                                 0],
                    [math.cos(alpha) * math.sin(beta), -math.sin(alpha), -math.cos(alpha) * math.cos(beta)],
                    [math.sin(alpha) * math.sin(beta), math.cos(alpha) , -math.sin(alpha) * math.cos(beta)]]
    
    # dR/dbeta   
    diff_R_beta = [[                  -math.sin(beta),  0,                   math.cos(beta)],
                   [ math.sin(alpha) * math.cos(beta),  0, math.sin(alpha) * math.sin(beta)],
                   [-math.cos(alpha) * math.cos(beta),  0, -math.cos(alpha) * math.sin(beta)]]
    
    M = xmk.getRobotManipulability1(angles, R_base_body[0:3, 0:3], grounded_feet, feetpos_body, diff_R_alpha, diff_R_beta)
    
    # After robot settles down...
    if t > t_wait:
        
        ct = t - t_wait
        
        # Manual override
        # x_dot = 0
        # y_dot = 0
        # z_dot = 0
        # # alpha_dot = 0
        # beta_dot = 0
        
        # Update Body Pose
        x += x_dot * dt 
        y += y_dot * dt
        z += z_dot * dt
        alpha += alpha_dot * dt
        beta += beta_dot * dt
        
        # z = 0.05 * (1 - math.cos(omg_body * (ct))) / 2 - 0.1
        # alpha = 0.15 * (1 - math.cos(omg_body * (ct))) / 2 + 0.1
        
        # z_claws_left = 0.35 * (1 - math.cos(omg_claws[0] * (ct - 10))) / 2 
        # y_claws_left = 0.05 * (1 - math.cos(omg_claws[0] * (ct - 10))) / 2
        
        # z_claws_right = 0.35 * (1 - math.cos(omg_claws[1] * (ct - 20))) / 2 
        # y_claws_right = 0.05 * (1 - math.cos(omg_claws[1] * (ct - 20))) / 2 
        
        # P_base[2,0] = z + P_drum_0[2,0] + 0.1 + z_claws_left
        # P_base[2,1] = z + P_drum_0[2,1] + 0.1 + z_claws_right
        
        # P_base[0,0] = P_drum_0[0,0] - y_claws_left
        # P_base[0,1] = P_drum_0[0,1] - y_claws_right
        
        # print(z, math.sin(omg * ct) >= 0)
        
        time_period = 1000
        
        if not control: [x, y, z, alpha, beta, gamma] , P_base = anm.getSplinePoint(ct, time_period, pose_count, bodypose_dance, P_dance, True)
        
    
    # X-axes
    plotF[0][t] = t
    plotT[0][t] = t
    plotM[0][t] = t
    plotTEq[0][t] = t
    plotTmax[0][t] = t
    plotPosition[0][t] = t
    
    # For plotting Forces
    for i in range(len(f)):
        plotF[i+1][t] = f[i] 
    plotF[7][t] = f_avg
    
    # For plotting Torques
    for i in range(len(torques_p[0])):
        plotT[i+1][t] = torques_p[0][i] 
    
    # For plotting Manipulability objective function
    plotM[1][t] = M
    
    # For plotting Equivalent Torque
    plotTEq[1][t] = np.linalg.norm(torques_p[0])
    
    # For plotting Maximum Torque
    plotTmax[1][t] = np.max(torques_p[0])
    
    # For plotting Position
    plotPosition[1][t], plotPosition[2][t], plotPosition[3][t], plotPosition[4][t], plotPosition[5][t]  = x, y, z, alpha, beta

    loopTime = time.perf_counter() - tStart
    time.sleep(max(0,dt-loopTime))

# Plot Forces
plt.figure(1)
for i in range(len(plotF) - 1):
    plt.plot(plotF[0], plotF[i+1], c=colors_endForce[i])
    
# Plot Torques
plt.figure(2)
for i in range(len(plotT) - 2):
    plt.plot(plotT[0], plotT[i+1], c=colors_jointTorque[i])    

# Plot Manipulability objective function
plt.figure(3)
plt.plot(plotM[0], plotM[1], c=colors_jointTorque[0])
ax = plt.gca()
ax.set_ylim([0, 3e-6]) 

# Plot Equivalent Torque
plt.figure(4)
plt.plot(plotTEq[0], plotTEq[1], c=colors_jointTorque[1])
plt.plot(plotPosition[0], plotPosition[3], c=colors_jointTorque[5])

# Plot Maximum Torque
plt.figure(5)
plt.plot(plotTmax[0], plotTmax[1], c=colors_jointTorque[2])

# Plot position
plt.figure(6)
for i in range(len(plotPosition) - 1):
    plt.plot(plotPosition[0], plotPosition[i+1], c=colors_endForce[i+1])    

plt.show()
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
Sensing and actuation are in 1,3,2,4,6 order
Processing is in 1,2,3,4,5,6 order

 [[0.5,  0.5,  0.1, 0.1, -0.3, -0.3],    # Left 15
[ 0.43, -0.2,  0.45, -0.45,  0.42, -0.42],
[ 0.1, 0.15, -0.2, -0.2, -0.2, -0.2]]
                    
[-0.05, -0.1, -0.05, 0, -0.1, -0.4] # Left 15                    

"""
