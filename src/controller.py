import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from simple_pid import PID
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String, Bool, Float32, Float64, Float32MultiArray

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

import time


def pi_clip(angle):
    """function to map angle error values between [-pi, pi]"""
    if angle > 0:
        if angle > math.pi:
            return angle - 2*math.pi
    else:
        if angle < -math.pi:
            return angle + 2*math.pi
    return angle

class vehicleController():

    pid_angle = PID(-2, -0.5, -0.0, setpoint=0.0, output_limits=(-5, 5))
    pid_angle.error_map = pi_clip #function to map angle errror values between -pi and pi
    pid_velocity = PID(-20, -10, -0.0, setpoint=0.0, output_limits=(0, 2))

    def __init__(self):

        self.dt = 0.1
        self.rate = rospy.Rate(30)
        self.target_throttle = 0.33
        self.lane_orientation_sub = rospy.Subscriber('/rtabmap/localization_pose', PoseWithCovarianceStamped, self.control_callback)
        self.lane_orientation = 0.0
        self.ekf_x, self.ekf_y, self.ekf_heaading = 0.0, 0.0, 0.0

        # -------------------- PACMod setup --------------------
        self.gem_enable    = True
        self.pacmod_enable = True

        # GEM vehicle enable
        self.enable_sub = rospy.Subscriber('/pacmod/as_rx/enable', Bool, self.pacmod_enable_callback)
        # self.enable_cmd = Bool()
        # self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second



    def control_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        [phi, theta, psi] = self.quaternion_to_euler(q.x, q.y, q.z, q.w)
        self.ekf_x, self.ekf_y = p.x, p.y
        self.ekf_heaading = psi    # heading in degrees world frame

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return [roll, pitch, yaw]
        


    # PACMod enable callback function
    def pacmod_enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def compute_lateral_error(self, image):
        pass

    def longititudal_PID_controller(self, lateral_error):

        target_velocity = self.pid_velocity(lateral_error, dt=self.dt)
        pass

    # Task 3: Lateral Controller (Pure Pursuit)
    def lateral_PID_controller(self, lane_orentation):
        steering_angle = self.pid_angle(lane_orentation, dt = self.dt) #radians
        return steering_angle

    
    # Start PACMod interface
    def start_pacmod(self):
        
        # while not rospy.is_shutdown():

            if(self.pacmod_enable == True):
                
                print(self.ekf_x, self.ekf_y, self.ekf_heaading) 
                    
            self.rate.sleep()


def execute():
    rospy.init_node('pacmod_control_node', anonymous=True)
    controller = vehicleController()
    while not rospy.core.is_shutdown():
        try:
            controller.start_pacmod()
        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    execute()