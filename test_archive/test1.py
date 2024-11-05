# no longer applicable as all functions have been updated

from Yuna import Yuna
import time
import numpy as np 

yuna = Yuna(real_robot_control=0, pybullet_on=1)
yuna.env.camerafollow = False
time.sleep(1)
yuna.env.load_wall()

# clinb wall on left side with three left legs

front_x, front_y = 0.51589, 0.23145
middle_x, middle_y = 0.0575, 0.5125
back_x, back_y = 0.45839, 0.33105
front_y_diff = middle_y - front_y
back_y_diff = middle_y - back_y

raise_h = 0.3
tuck_in_dist = 0.1
align_front_y_dist = 0.25
align_back_y_dist = 0.3


##### PHASE 1: MOST CONSTRAINED POSITION (LEGS IS TUCKED IN AS CLOSE AS POSSIBLE)
### V1: raise and align legs one by one, tuck in using a separate step
# # align back legs
# yuna.raise_leg(leg_index=4, dz=raise_h)
# yuna.move_leg(leg_index=4, dx=align_front_y_dist, dy=back_y_diff, dz=-raise_h)
# yuna.raise_leg(leg_index=5, dz=raise_h)
# yuna.move_leg(leg_index=5, dx=align_front_y_dist, dy=-back_y_diff, dz=-raise_h)

# # align front legs
# yuna.raise_leg(leg_index=0, dz=raise_h)
# yuna.move_leg(leg_index=0, dx=-align_back_y_dist, dy=front_y_diff, dz=-raise_h)
# yuna.raise_leg(leg_index=1, dz=raise_h)
# yuna.move_leg(leg_index=1, dx=-align_back_y_dist, dy=-front_y_diff, dz=-raise_h)

# # tuck in legs (front)
# yuna.raise_leg(leg_index=0, dz=raise_h)
# yuna.move_leg(leg_index=0, dx=0, dy=-tuck_in_dist, dz=-raise_h)
# yuna.raise_leg(leg_index=1, dz=raise_h)
# yuna.move_leg(leg_index=1, dx=0, dy=tuck_in_dist, dz=-raise_h)


### V2: align and tuck in at the same time
# # front
# yuna.raise_leg(leg_index=0, dz=raise_h)
# yuna.move_leg(leg_index=0, dx=-align_front_y_dist, dy=front_y_diff-tuck_in_dist, dz=-raise_h)
# yuna.raise_leg(leg_index=1, dz=raise_h)
# yuna.move_leg(leg_index=1, dx=-align_front_y_dist, dy=-front_y_diff+tuck_in_dist, dz=-raise_h)

# # back
# yuna.raise_leg(leg_index=4, dz=raise_h)
# yuna.move_leg(leg_index=4, dx=align_back_y_dist, dy=back_y_diff-tuck_in_dist, dz=-raise_h)
# yuna.raise_leg(leg_index=5, dz=raise_h)
# yuna.move_leg(leg_index=5, dx=align_back_y_dist, dy=-back_y_diff+tuck_in_dist, dz=-raise_h)

### V3: move two legs at the same time (raise + align + tuck in right side)
# front left, back right
raise_leg16 = np.zeros((3,6))
raise_leg16[:, 0] = [0, 0, raise_h]
raise_leg16[:, 5] = [0, 0, raise_h]
yuna.move_legs(raise_leg16)
align_leg16 = np.zeros((3,6))
align_leg16[:, 0] = [-align_front_y_dist, front_y_diff, -raise_h]
align_leg16[:, 5] = [align_back_y_dist, -back_y_diff+tuck_in_dist, -raise_h]
yuna.move_legs(align_leg16)

# front right, back left
raise_leg25 = np.zeros((3,6))
raise_leg25[:, 1] = [0, 0, raise_h]
raise_leg25[:, 4] = [0, 0, raise_h]
yuna.move_legs(raise_leg25)
align_leg25 = np.zeros((3,6))
align_leg25[:, 1] = [-align_front_y_dist, -front_y_diff+tuck_in_dist, -raise_h]
align_leg25[:, 4] = [align_back_y_dist, back_y_diff, -raise_h]
yuna.move_legs(align_leg25)


##### PHASE 2: WALL TRANSITION
def raise_to_wall(leg_index):
    for i in range(4): # to move up in a straight line to avoid hitting the wall
        yuna.raise_leg(leg_index)
    yuna.move_leg(leg_index=leg_index, dx=0, dy=0.075, dz=0)

raise_to_wall(0)
raise_to_wall(2)
raise_to_wall(4) # when all three left legs are raised, body frame of robot will change? then pose for all the legs need to be adjusted?

time.sleep(1e5)