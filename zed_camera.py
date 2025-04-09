import pyzed.sl as sl
import numpy as np
import time
import pybullet as p

np.set_printoptions(precision=4, suppress=True)

class ZedCamera:
    def __init__(self):
        self.camera = sl.Camera()
        # Get the distance between the center of the camera and the left eye
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 or HD1200 video mode (default fps: 60)
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD # Right handed, z-up, x-forward (ROS) -> if change this need change how self.tx is obtained and how tranform_pose is calculated
        self.init_params.coordinate_units = sl.UNIT.METER  # Set units in meters

        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit()
 
        self.runtime_parameters = sl.RuntimeParameters()

        # https://www.stereolabs.com/docs/positional-tracking/coordinate-frames
        # self.tx = self.camera.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.get_translation().get()[1] # distance between left and right eye -> [1] because using RIGHT_HANDED_Z_UP_X_FWD
        # self.translation_left_to_center = self.tx * 0.5

        # Enable positional tracking with default parameters
        py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
        tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
        err = self.camera.enable_positional_tracking(tracking_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Enable positional tracking : "+repr(err)+". Exit program.")
            self.camera.close()
            exit()

    def close(self):
        self.camera.close()

    def get_robot_frame_wrt_world(self):
        # W_T_R (robot_wrt_world) = W_T_C (camera_wrt_world) * C_T_C (camera_wrt_camera) * C_T_R (robot_wrt_camera)
        camera_left_eye_to_camera_center = 0.06 # 0.06m
        cam_to_robot_base_height = 0.2 # doesnt matter cuz it's gonna cancel out anyways
        robot_base_to_ground_height = 0.145

        camera_pose = sl.Pose() # C_T_C
        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                
            self.camera.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD) # pose of the Camera Frame located on the left eye. To get the movement of the center of the camera, you need to add a rigid transform. 

            camera_wrt_world = sl.Transform()
            camera_wrt_world.set_identity()
            camera_wrt_world[1, 3] = camera_left_eye_to_camera_center
            camera_wrt_world[2, 3] = cam_to_robot_base_height + robot_base_to_ground_height

            robot_wrt_camera = sl.Transform()
            robot_wrt_camera.set_identity()
            robot_wrt_camera[1, 3] = - camera_left_eye_to_camera_center
            robot_wrt_camera[2, 3] = - cam_to_robot_base_height 

            robot_wrt_world = camera_wrt_world * camera_pose.pose_data(sl.Transform()) * robot_wrt_camera
            # print("robot_wrt_world:\n", robot_wrt_world)

            result_transform = sl.Transform()
            result_transform.init_matrix(robot_wrt_world)
            result_pose = sl.Pose()
            result_pose.init_transform(result_transform)

            result_orn = sl.Orientation()
            result_orn = np.array(result_pose.get_orientation(result_orn).get())
            result_pos = sl.Translation()
            result_pos = np.array(result_pose.get_translation(result_pos).get())
            # print("result_orn:\n", result_orn)
            # print("result_pos:\n", result_pos)

            return result_pos, result_orn

    def get_camera_frame(self):
        # default Camera Frame located on the left eye. 
        # example code at https://www.stereolabs.com/docs/positional-tracking/using-tracking
        camera_pose = sl.Pose()
        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                
            self.camera.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD) 
            # in world frame -> aka current pose of camera wrt initial pose (upon initialization)
            raw_orientation = sl.Orientation()
            raw_orientation = np.array(camera_pose.get_orientation(raw_orientation).get()) # in quaternion
            rotation_matrix = np.array(p.getMatrixFromQuaternion(raw_orientation)).reshape(3, 3)
            print("raw_rotation_matrix:\n", rotation_matrix)
            
            raw_translation = sl.Translation()
            raw_translation = np.array(camera_pose.get_translation(raw_translation).get())
            print("raw_translation:\n", raw_translation)

            return raw_orientation, raw_translation


if __name__ == "__main__":
    zed_camera = ZedCamera()

    try:
        while True:
            zed_camera.get_camera_frame()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        zed_camera.close()