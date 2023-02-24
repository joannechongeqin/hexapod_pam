import numpy as np
from robot_setup.yunaKinematics import HexapodKinematics

def rot(pos, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pos_ = np.matmul(rot_z, pos)
    return pos_

def trans(pos, distance, angle):
    pos_ = pos.copy()
    pos_[0] += distance * np.cos(angle)
    pos_[1] += distance * np.sin(angle)
    return pos_

def hebi2bullet(jointspace_command2hebi):
    # reorder the jointspace command so that it can work in pybullet
    jointspace_command2bullet = jointspace_command2hebi[[0,1,2,6,7,8,12,13,14,3,4,5,9,10,11,15,16,17,]].copy()# reshaped the IK result: 123456->135246
    return jointspace_command2bullet
    
def solveIK(workspace_command):
    xmk = HexapodKinematics()
    jointspace_command2hebi = xmk.getLegIK(workspace_command)
    jointspace_command2bullet = hebi2bullet(jointspace_command2hebi)
    return jointspace_command2bullet, jointspace_command2hebi

def solveFK(jointspace_command2hebi):
    xmk = HexapodKinematics()
    workspace_command = xmk.getLegPositions(np.array([jointspace_command2hebi]))
    return workspace_command