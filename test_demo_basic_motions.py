<<<<<<< Updated upstream:test_demo_basic_motions.py
from Yuna import Yuna
import time
import numpy as np

yuna = Yuna(real_robot_control=0, pybullet_on=1)
yuna.env.camerafollow = False
# yuna.env.load_wall()


time.sleep(2)
# move one leg
leg1_raise = np.zeros((3,6))
leg1_raise[2, 0] = 0.2
leg1_down = np.zeros((3,6))
leg1_down[2, 0] = -0.2
yuna.move_legs_by_pos_in_world_frame([leg1_raise, leg1_down])
time.sleep(2)
# yuna.move_legs_by_pos_in_world_frame([leg1_raise, leg1_down])
time.sleep(2)

# move two legs simultaneously
leg26_raise = np.zeros((3,6))
leg26_raise[2, 1] = 0.2
leg26_raise[2, 5] = 0.2
leg26_down = np.zeros((3,6))
leg26_down[:, 1] = [0.1, 0.2, -0.2]
leg26_down[:, 5] = [-0.1, 0.2, -0.2]
leg26_down2 = np.zeros((3,6))
leg26_down2[:, 1] = [-0.1, -0.2, -0.2]
leg26_down2[:, 5] = [0.1, -0.2, -0.2]
yuna.move_legs_by_pos_in_world_frame([leg26_raise, leg26_down])
time.sleep(1)
yuna.move_legs_by_pos_in_world_frame([leg26_raise, leg26_down2])
time.sleep(5)

# rotate
yuna.rotx_body(10, move=True)
yuna.rotx_body(-20, move=True)
yuna.rotx_body(10, move=True)
time.sleep(2)

# translate
yuna.trans_body(0.1, 0, 0, move=True)
yuna.trans_body(-0.1, 0, 0, move=True)
yuna.trans_body(0, 0.1, 0, move=True)
yuna.trans_body(0, -0.1, 0, move=True)
yuna.trans_body(0, 0, 0.1, move=True)
yuna.trans_body(0, 0, -0.1, move=True)

time.sleep(10)
=======
from Yuna import Yuna
import time
import numpy as np

yuna = Yuna(real_robot_control=0, pybullet_on=1)
yuna.env.camerafollow = False
# yuna.env.load_wall()


time.sleep(2)
# move one leg
leg1_raise = np.zeros((3,6))
leg1_raise[2, 0] = 0.2
leg1_down = np.zeros((3,6))
leg1_down[2, 0] = -0.2
yuna.move_legs_by_pos_in_world_frame([leg1_raise, leg1_down])
time.sleep(2)
# yuna.move_legs_by_pos_in_world_frame([leg1_raise, leg1_down])
time.sleep(2)

# move two legs simultaneously
leg26_raise = np.zeros((3,6))
leg26_raise[2, 1] = 0.2
leg26_raise[2, 5] = 0.2
leg26_down = np.zeros((3,6))
leg26_down[:, 1] = [0.1, 0.2, -0.2]
leg26_down[:, 5] = [-0.1, 0.2, -0.2]
leg26_down2 = np.zeros((3,6))
leg26_down2[:, 1] = [-0.1, -0.2, -0.2]
leg26_down2[:, 5] = [0.1, -0.2, -0.2]
yuna.move_legs_by_pos_in_world_frame([leg26_raise, leg26_down])
time.sleep(1)
yuna.move_legs_by_pos_in_world_frame([leg26_raise, leg26_down2])
time.sleep(5)

# rotate
yuna.rotx_body(10, move=True)
yuna.rotx_body(-20, move=True)
yuna.rotx_body(10, move=True)
time.sleep(2)

# translate
yuna.trans_body(0.1, 0, 0, move=True)
yuna.trans_body(-0.1, 0, 0, move=True)
yuna.trans_body(0, 0.1, 0, move=True)
yuna.trans_body(0, -0.1, 0, move=True)
yuna.trans_body(0, 0, 0.1, move=True)
yuna.trans_body(0, 0, -0.1, move=True)

time.sleep(10)
>>>>>>> Stashed changes:demo_basic_motions.py
