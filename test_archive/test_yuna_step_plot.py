from Yuna import Yuna
import time
import numpy as np

yuna = Yuna(real_robot_control=0, pybullet_on=1)
yuna.env.camerafollow = False

yuna.step(step_len=0.1, course=0, rotation=0)
yuna.env.plot_reaction_forces_and_torque(17)
yuna.env.plot_reaction_forces_and_torque(1)
yuna.env.plot_reaction_forces_and_torque(2)

time.sleep(1e5)
