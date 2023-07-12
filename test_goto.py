from Yuna import Yuna
import time

yuna = Yuna(real_robot_control=0, pybullet_on=1)
yuna.env.camerafollow = 1
#time.sleep(5)
yuna.step(step_len=0.1, course=0, rotation=0, steps=5)
yuna.stop()
time.sleep(1)
yuna.goto(dx=5, dy=0, dtheta=180)

time.sleep(1e5)