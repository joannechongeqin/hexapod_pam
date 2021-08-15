import numpy as np
from CPG.computeOffsets import computeOffsets
from Tools.constrainSE3 import *
from CPG.limitValue import *
from CPG.groundIK import *
# from CPG.supelliptRot import *
# from CPG.kuramoto import *
import copy
import math
import time
from matplotlib import pyplot as plt

pi = np.pi


dist_old = np.array([0.38144, -0.38144, 0.51249, -0.51249, 0.33105, -0.33105])

def CPG(cpg, t, dt):

    np.set_printoptions(precision=5)

    set_lastang = np.array([-pi / 2, pi / 2, -pi / 2, pi / 2, -pi / 2, pi / 2])

    shoulders1 = range(0, 18, 3)  # joint IDs of the shoulders
    shoulders2 = range(1, 18, 3)  # joint IDs of the second shoulder joints
    elbows = range(2, 3, 18)  # joint IDs of the elbow joints

    dynOffsetError = copy.deepcopy(np.zeros([3, 6]))

    if t > cpg['initLength']:
        [dynOffsetError, cpg] = computeOffsets(cpg, t, dt)

    # Scale
    dynOffsetError = dynOffsetError * cpg['scaling']

    y0 = cpg['dynOffset'][1, :]
    # cpg['cy'][cpg['CPGStance']] = y0[cpg['CPGStance']]
    cpg['cy'] = y0
    # print("cy = " ,cpg['cy'])


    # CPG Implmentation/Equations
    lambd = 3  # coupling strength
    gamma = 0.3  # force limit cycle to converge x dimension
    beta = 150  # force limit cycle to converge y dimension
    f = 1.5  # frequency

    cpg['a'] = np.reshape(cpg['a'], 6)
    if cpg['move']:

        H = ((cpg['x'][t, :] - cpg['cx']) / cpg['a']) ** 2 + ((cpg['y'][t, :] - cpg['cy']) / cpg['b']) ** 2
        w = np.ones(6) * 0.8 * np.pi * f
        w_swing = 1 * np.pi * f
        w_stance = 1 * np.pi * f

        phase = np.arctan2((cpg['y'][t, :] - cpg['cy']), (cpg['x'][t, :] - cpg['cx']))
        w_s = np.ones((6, 1)) * np.arctan2((cpg['y'][t, :] - cpg['cy']), (cpg['x'][t, :] - cpg['cx']))
        w += (lambd * np.sin(w_s - w_s.T - cpg['phi'])).sum(axis=0)
        # w[np.where(w < 0)] = 0

        cycle_x = w * (cpg['a'] * cpg['b']) * (cpg['y'][t, :] - cpg['cy']) / cpg['b'] ** 2
        cycle_y = - w * (cpg['a'] * cpg['b']) * (cpg['x'][t, :] - cpg['cx']) / cpg['a'] ** 2

        cycle_x = np.maximum(np.minimum(cycle_x, 30), -30)
        cycle_y = np.maximum(np.minimum(cycle_y, 30), -30)

        converge_x = gamma * (1 - H) * (cpg['x'][t, :] - cpg['cx']) / cpg['a'] ** 2
        converge_y = beta * (1 - H) * (cpg['y'][t, :] - cpg['cy']) / cpg['b'] ** 2

        dx = cycle_x + converge_x
        dy = cycle_y + converge_y

        cpg['dx'] = dx
        cpg['dy'] = dy

        # for value in range(6):
        #     truth = (cpg['CPGStanceDelta'].flatten())[value]
        #     if truth:
        #         cpg['xGr'][value] = cpg['x'][t, value]
        #
        # x_pos = abs(cpg['nomY'] - cpg['s1OffsetY'])
        #
        # dx_fact = 1 / (x_pos * np.reciprocal(np.cos(cpg['s1OffsetAngY'] + cpg['legs'][0, shoulders1])) ** 2)

        # dx_const = -1 * cpg['scaling'] * cpg['desired_speed'] * dx_fact
        # print(dx_const)
        truther = False
        cpg['updStab'] = np.logical_or(cpg['CPGStance'], (dy < 0))
        # updStab = cpg['CPGStance']
        # print('updStab',updStab)
        # print('dy[0] < 0', dy < 0)
        # print(cpg['cx']*180/pi)

    else:
        dx = 0
        dy = 0
        truther = True
        dx_const = 0
        updStab = np.logical_or(cpg['CPGStance'], np.array([False, False, False, False, False, False]))

    # test_ang = np.array([0, 0, -1.57,
    #                      0, 0, 1.57,
    #                      0, 0, -1.57,
    #                      0, 0, 1.57,
    #                      0, 0, -1.57,
    #                      0, 0, 1.57])
    # test_ang = np.reshape(test_ang[0:18], [1, 18])
    #
    # test_positions = cpg['xmk'].getLegPositions(test_ang)
    # print(test_positions)
    # [[0.51611  0.51611  0.0575   0.0575 - 0.45861 - 0.45861]
    #  [0.23158 - 0.23158  0.51276 - 0.51276  0.33118 - 0.33118]
    # [-0.2249 - 0.2249 - 0.2249 - 0.2249 - 0.2249 - 0.2249]]
    # ang = cpg['xmk'].getLegIK(test_positions)
    # print(ang)

    # Calculate dynOffsetInc
    cpg['pid'].update(dt, dynOffsetError, cpg['updStab'])
    cpg['dynOffset'] = cpg['pid'].getCO()
    cpg['dynOffsetInc'] = cpg['pid'].getDeltaCO()

    # print('offset',cpg['dynOffset'][1,:])
    # print('Increas',cpg['dynOffsetInc'][1,:])
    # print(cpg['CPGStance'])

    # Integrate dx & dy to produce joint commands

    cpg['x'][t + 1, :] = cpg['x'][t, :] + dx * dt  # + cpg['dxoffset']*5  # + cpg['dynOffsetInc'][0,:]
    # cpg['xGr'] = cpg['xGr'] + dx_const * dt # + cpg['dynOffsetInc'][0,:]
    cpg['y'][t + 1, :] = cpg['y'][t, :] + dy * dt + cpg['dynOffsetInc'][1, :]



    # Command CPG-generated values to joints
    yOut = cpg['y'][t + 1, :] #+ cpg['dynOffsetInc'][1, :]  # *cpg['CPGStance']
    xOut = np.zeros([1, 6])

    SignBack = -1 if (cpg['direction'] == 'backwards') else 1
    SignLeft = -1 if (cpg['direction'] == 'left') else 1
    SignRight = -1 if (cpg['direction'] == 'right') else 1


    # plt.figure("dx")
    # if (t % 10) == 0:
    #     if cpg['y'][t + 1, 0] < 0:
    #         cpg_dx.append(dx[0])
    #     plt.plot(cpg_dx)
    #     plt.pause(0.00001)

    # for value in range(6):
    #     truth = cpg['CPGStance'][value]
    #     if truth:
    #         xOut[:, value] = cpg['xGr'][value]
    #     else:
    #         if value % 2 == 0:
    #             xOut[:, value] = SignLeft * SignBack * cpg['x'][t + 1, value]
    #         else:
    #             xOut[:, value] = SignRight * SignBack * cpg['x'][t + 1, value]

    for value in range(6):
        if value % 2 == 0:
            xOut[:, value] = SignLeft * SignBack * cpg['x'][t + 1, value]
        else:
            xOut[:, value] = SignRight * SignBack * cpg['x'][t + 1, value]

    # cpg['legs'][0,0:18:3] = limitValue((cpg['shouldersCorr'] * xOut), pi/2 * cpg['scaling'])
    cpg['legs'][0, 0:18:3] = limitValue((cpg['nomOffset'][0, :] + cpg['shouldersCorr'] * xOut), pi / 2 * cpg['scaling'])
    cpg['legs'][0, 1:19:3] =cpg['shouldersCorr'] * np.maximum(y0, yOut) + cpg['nomOffset'][1, :]
    # print('shoulders', np.maximum(y0, yOut))

    #  cpg['shouldersCorr'] helps correct the output with differernt side of motor
    # cpg['nomOffset'])  is the rest ppositon of the robot

    # JOINT 3 - FOR WALKING TRIALS
    cpg['legs'][0, 0:18:3] = cpg['legs'][0, 0:18:3]/10 # cpg['scaling']
    # print(cpg['legs'][0, 0:18:3])
    # print('a',cpg['a'])
    cpg['legs'][0, 1:19:3] = cpg['legs'][0, 1:19:3]/10 # cpg['scaling']

    # print(cpg['legs'][0, 1],cpg['legs'][0, 16])

    I = (np.logical_or(cpg['CPGStance'], dy < 0))  # the leg on the ground


        #cpg['nomY']  # coordinate(y) of the end effector of 6 legs

    indicies = []    # the legs are falling and on the ground
    for index in range(6):
        if truther:     # moving : truther = false
            truth = I[index]
        else:
            truth = I[index]
        if truth:
            indicies.append(index)

    z = np.array([-math.pi/2, math.pi/2, -math.pi/2, math.pi/2, -math.pi/2, math.pi / 2])
    # print("inside newCPG",cpg["legs"][0,0:3])
    a_0 = cpg['a'] / cpg['scaling']
    c_x = cpg['cx'] / cpg['scaling']

    if cpg['t'] > (cpg['initLength'] + 50) and cpg['direction'] == 'forward':

        dist = cpg['foothold_offsetY']
        angs = groundIK(cpg,dist,indicies)
        for index in indicies:
            cpg['legs'][0,2+index*3] = angs[2+index*3]  #+(cpg['dynOffset'][2,index]/cpg['scaling'])


    else:
        for i in range(6):
            if cpg['legs'][0, 1 + 3 * i] != 0:
                z[i] = (-1) ** (i + 1) * math.pi / 2
        cpg['legs'][0, 2] = z[0]
        cpg['legs'][0, 5] = z[1]
        cpg['legs'][0, 8] = z[2]
        cpg['legs'][0, 11] = z[3]
        cpg['legs'][0, 14] = z[4]
        cpg['legs'][0, 17] = z[5]

    cpg['legs'] = np.reshape(cpg['legs'][0:18], [1, 18])

    legs = np.copy(cpg['legs'])
    # print(cpg['legs'])

    if cpg['t'] > (cpg['initLength'] + 100):
        positions = cpg['xmk'].getLegPositions(cpg['legs'])
    else:
        positions = np.array([[0.50588, 0.63498, 0.1406, -0.00552, -0.47454, -0.54181],
                              [0.2559, -0.25935, 0.60987, -0.5175, 0.31018, -0.42041],
                              [-0.2249, -0.00234, -0.02791, -0.2249, -0.2249, -0.00273]])

    # print(cpg['legs'][0][0])

    return cpg,positions
