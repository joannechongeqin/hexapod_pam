import hebi
import math
import numpy as np

class HexapodAnimatronics(object):
    
    def getSplinePoint(self, real_time, time_period, pose_count, target_frames1, target_frames2, cyclic):
    
        # Catmull-Rom Spline
        # "pose_count" must be an integer giving the length of "target_frames1" and "target_frames2"
        # "real_time" is the time stamp of motion and "time_period" is the duration of time between two poses
    
        # Normalise t based on time duration for every interval
        t = real_time / time_period
        
        # For looping to first frame after completing last frame
        if cyclic == True:
            
            p1 = int((t // 1) % pose_count)
            p2 = int((p1 + 1) % pose_count)
            p3 = int((p2 + 1) % pose_count)
            if p1 >= 1:
                p0 = int(p1 - 1)
            else:
                p0 = int(pose_count - 1)
        
        # For pausing after list of frames ends. Note: By definition, 1st and Last frames will never be achieved. Motion will begin on second frame and end on last frame
        else:      
            p0 = int(t // 1)
            p1 = int((p0 + 1) % pose_count)
            p2 = int((p1 + 1) % pose_count)
            p3 = int((p2 + 1) % pose_count)
            
            # Finding out if the list has ended
            if p0 >= pose_count - 3:
                print("Out of range, stopping")
                
                # Return final pose in sequence, the second-last frame.
                return target_frames1[pose_count - 2], target_frames2[pose_count - 2]

        # Normalise time to fractional value
        t = t - (t // 1)

        # Special polynomials of Catmull-Rom Spline
        q = np.zeros([4,1])
        q[0] = -(t**3) + 2 * (t**2) - t
        q[1] = 3 * (t**3) - 5 * (t**2) + 2 
        q[2] = -3 * (t**3) + 4 * (t**2) + t
        q[3] = t**3 - t**2
    
        # Output frames for each set of frames
        spline_frame1 = 0.5 * (target_frames1[p0] * q[0] + target_frames1[p1] * q[1] + target_frames1[p2] * q[2] + target_frames1[p3] * q[3]) 
        spline_frame2 = 0.5 * (target_frames2[p0] * q[0] + target_frames2[p1] * q[1] + target_frames2[p2] * q[2] + target_frames2[p3] * q[3]) 
    
        return spline_frame1, spline_frame2
    
'''Built upon the tutorial by javidx9's channel on YouTube: https://www.youtube.com/watch?v=9_aJGUTePYo&t=1127s'''
    