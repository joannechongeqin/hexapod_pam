from Yuna import Yuna
import time

yuna = Yuna()

#example test code for controlling yuna
yuna.walk(stride=-0.4, angle=0, step=3)#math.pi/2
yuna.stop()
yuna.walk(stride=0.05, angle=0,step=5)
yuna.walk(stride=0.3, angle=0,step=5)
yuna.walk(stride=0.1, angle=180,step=8)
yuna.walk(stride=0.1, angle=60,step=8)
yuna.walk(stride=0.1, angle=120,step=8)
yuna.walk(stride=0.1, angle=270,step=8)
yuna.stop()
yuna.turn(deg_per_step=30, step=6)
yuna.turn(deg_per_step=15, step=5)
yuna.walk(stride=0.1, angle=270,step=8)
yuna.turn(deg_per_step=-30, step=8)
yuna.turn(deg_per_step=5, step=8)
yuna.stop()
time.sleep(1e6)