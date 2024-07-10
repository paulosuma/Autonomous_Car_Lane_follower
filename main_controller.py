# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import alvinxy as axy # Import AlvinXY transformation module
import rospy

# GEM Sensor Headers
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool, Float32, Float64
from geometry_msgs.msg import PoseWithCovarianceStamped
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva
from simple_pid import PID

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
    pid_velocity = PID(-20, -10, -0.0, setpoint=0.0, output_limits=(0.3, 0.48))

    def __init__(self):

        self.dt = 0.1
        self.rate = rospy.Rate(30)
        self.target_throttle = 0.33
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.86 # meters

        self.lane_orientation_sub = rospy.Subscriber('/lane_orientation', Float32, self.control_callback)
        self.lane_orientation = 0.0

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.curr_speed      = 0.0

        self.desired_speed = 1.5  # m/s, reference speed
        self.max_accel     = 0.48 # % of acceleration

        # self.pose_subscriber = rospy.Subscriber("/odometry/filtered", Odometry, self.ekf_callback)
        self.pose_subscriber = rospy.Subscriber("/rtabmap/localization_pose", PoseWithCovarianceStamped, self.ekf_callback)
        self.ekf_x, self.ekf_y, self.ekf_heading = 0.0, 0.0, 0.0

        self.prev_alpha = 0.0


        # -------------------- Load Map --------------------
        filepath = "/home/paulosuma/Documents/SafeAuto/mp-release-23fa-main/src/gem_Project/output_odom_data.csv"
        self.center_lane = np.loadtxt(filepath)

        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = True

        # GEM vehicle enable, publish once
        self.enable_pub = rospy.Publisher('/pacmod/as_rx/enable', Bool, queue_size=1)
        self.enable_cmd = Bool()
        self.enable_cmd.data = False

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



    # Read lane orientation from student vision
    def control_callback(self, msg):
        self.lane_orientation = msg.data #radians

    def speed_callback(self, msg):
        self.curr_speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    # PACMod enable callback function
    def pacmod_enable_callback(self, msg):
        self.pacmod_enable = msg.data

    #ekf callback function
    def ekf_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        [phi, theta, psi] = self.quaternion_to_euler(q.x, q.y, q.z, q.w)
        self.ekf_x, self.ekf_y = p.x, p.y
        self.ekf_heading = psi #radians

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

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr


    ####################### Get Gem State from Visual Odometry #######################
    def get_gem_state_visual_odometry(self):
        
        local_x_curr, local_y_curr = self.ekf_x, self.ekf_y
        curr_yaw = self.ekf_heading

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)
        print(round(curr_x, 3), round(curr_y, 3))
        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)



    ####################### CONTROLLERS #######################
    def compute_lateral_error(self, image):
        pass

    # Pid controller for velocity
    def longititudal_PID_controller(self, vel_error):
        target_velocity = self.pid_velocity(vel_error, dt=self.dt)
        pass

    # Lateral Pid Controller
    def lateral_PID_controller_student_vision(self, lane_orentation):
        steering_angle = self.pid_angle(lane_orentation, dt = self.dt) #radians
        return steering_angle
    
    # -------------------- PID Controller for Steering --------------------
    def lateral_PID_controller(self, curr_x, curr_y, curr_yaw):
        centerline_coords = self.center_lane
        curr_point = np.array([curr_x, curr_y])
        euclidean_dist = np.linalg.norm(centerline_coords-curr_point, axis=1)
        closest_idx = np.argsort(euclidean_dist)[:3]
        closest_points = centerline_coords[closest_idx]
        print("Closest points are", closest_points)

        for point in closest_points:
            P_T_G = np.array([point[0], point[1], 1])
            Homo_B_G = np.array([[math.cos(curr_yaw), -math.sin(curr_yaw), curr_x], 
                                [math.sin(curr_yaw), math.cos(curr_yaw), curr_y], 
                                [0, 0, 1]])
            P_T_B = np.matmul(np.linalg.inv(Homo_B_G), P_T_G)
            if P_T_B[0] > 0: break

        angle_error = math.atan2(P_T_B[1], P_T_B[0])
        cross_track_error = (math.sqrt(P_T_B[1]**2+P_T_B[0]**2)*math.sin(angle_error))
        print("cross track error: ", cross_track_error)
        target_steering = self.pid_angle(angle_error, dt=self.dt)
        return target_steering


    # -------------------- Pure Pursuit Steering Controller --------------------
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw):
        centerline_coords = self.center_lane
        curr_point = np.array([curr_x, curr_y])
        euclidean_dist = np.linalg.norm(centerline_coords-curr_point, axis=1)
        closest_idx = np.argsort(euclidean_dist)[:3]
        closest_points = centerline_coords[closest_idx]

        for point in closest_points:
            P_T_G = np.array([point[0], point[1], 1])
            Homo_B_G = np.array([[math.cos(curr_yaw), -math.sin(curr_yaw), curr_x], 
                                [math.sin(curr_yaw), math.cos(curr_yaw), curr_y], 
                                [0, 0, 1]])
            P_T_B = np.matmul(np.linalg.inv(Homo_B_G), P_T_G)
            if P_T_B[0] > 0: break

        max_ld = 10
        alpha = math.atan2(P_T_B[1], P_T_B[0])
        cross_track_error = (math.sqrt(P_T_B[1]**2+P_T_B[0]**2)*math.sin(alpha))
        print("curr point: ", curr_x, curr_y)
        print("target point: ", P_T_B[0], P_T_B[1])
        # print("cross track error: ", cross_track_error)
        steering_diff = abs(alpha-self.prev_alpha)
        self.prev_alpha = alpha
        ld = max_ld*math.cos(steering_diff) 
        target_steering = math.atan((2*self.wheelbase*math.sin(alpha))/ld)

        return target_steering

    
    # Start PACMod interface
    def start_pacmod(self):

        if(self.pacmod_enable == True):
            
            if (self.gem_enable == False):

                # ---------- Enable PACMod ----------

                # enable forward gear
                self.gear_cmd.ui16_cmd = 3

                # enable brake
                self.brake_cmd.enable  = True
                self.brake_cmd.clear   = False
                self.brake_cmd.ignore  = False
                self.brake_cmd.f64_cmd = 0.0

                # enable gas 
                self.accel_cmd.enable  = True
                self.accel_cmd.clear   = False
                self.accel_cmd.ignore  = False
                self.accel_cmd.f64_cmd = 0.0

                self.gear_pub.publish(self.gear_cmd)
                print("Foward Engaged!")

                self.turn_pub.publish(self.turn_cmd)
                print("Turn Signal Ready!")
                
                self.brake_pub.publish(self.brake_cmd)
                print("Brake Engaged!")

                self.accel_pub.publish(self.accel_cmd)
                print("Gas Engaged!")

                self.gem_enable = True

            else: 
                # steering PID, lane orientation from studentvision.py
                # steering_angle = self.lateral_PID_controller_student_vision(self.lane_orientation)

                # steering angle from center lane
                curr_x, curr_y, curr_yaw = self.get_gem_state_visual_odometry()
                steering_angle = self.lateral_PID_controller(curr_x, curr_y, curr_yaw)

                # steering angle Pure Pursuit
                # curr_x, curr_y, curr_yaw = self.get_gem_state_from_GPS()
                # steering_angle = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw)

                if (steering_angle <= math.pi/4 and steering_angle >= -math.pi/4):
                    self.turn_cmd.ui16_cmd = 1
                elif(steering_angle > math.pi/4):
                    self.turn_cmd.ui16_cmd = 2 # turn left
                else:
                    self.turn_cmd.ui16_cmd = 0 # turn right

                ############## Acceleration #################
                print("current speed: ", self.curr_speed)
                vel_error = self.desired_speed-self.curr_speed
                accel = self.longititudal_PID_controller(vel_error)
                print("Current acceleration: ", accel)
                self.accel_cmd.f64_cmd = accel

                ############## Steering #####################
                self.steer_cmd.angular_position = steering_angle
                print("steering angle ", self.steer_cmd.angular_position)

                self.accel_pub.publish(self.accel_cmd)
                self.steer_pub.publish(self.steer_cmd)
        
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