                                                               
# Credit to Hang Cui (hangcui3@illinois.edu) for writing most part of the code
# Pure pursuit controller

from __future__ import print_function

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# Import Stop Sign Detector
# from vehicle_vision.stop_sign_detector import StopSignDetector

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

GEM_CAR = False

def pi_clip(angle):
    """function to map angle error values between [-pi, pi]"""
    if angle > 0:
        if angle > math.pi:
            return angle - 2*math.pi
    else:
        if angle < -math.pi:
            return angle + 2*math.pi
    return angle

class PID(object):

    def __init__(self, kp, ki, kd, wg=None):

        self.iterm  = 0
        self.last_t = None
        self.last_e = 0
        self.kp     = kp
        self.ki     = ki
        self.kd     = kd
        self.wg     = wg
        self.derror = 0

    def reset(self):
        self.iterm  = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):

        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        if abs(e - self.last_e) > 0.5:
            de = 0

        self.iterm += e * (t - self.last_t)

        # take care of integral winding-up
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        self.last_e = e
        self.last_t = t
        self.derror = de

        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de


class OnlineFilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted


class PurePursuit(object):
    
    def __init__(self):

        self.rate       = rospy.Rate(10)

        self.look_ahead = 4
        self.wheelbase  = 1.75 # meters
        self.offset     = 0.86 - 0.46 # meters

        self.gnss_sub   = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        self.lat        = 0.0
        self.lon        = 0.0
        self.latflag = False
        self.lonflag = False
        self.olat       = 40.0928563
        self.olon       = -88.2359994
        self.lon_wp_x, self.lat_wp_y = 0.0, 0.0

        self.enable_sub = rospy.Subscriber("/pacmod/as_tx/enable", Bool, self.enable_callback)

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        # self.pose_subscriber = rospy.Subscriber("/odometry/filtered", Odometry, self.ekf_callback)
        self.pose_subscriber = rospy.Subscriber("/rtabmap/localization_pose", PoseWithCovarianceStamped, self.ekf_callback)
        self.ekf_x, self.ekf_y, self.ekf_heading = 0.0, 0.0, 0.0

        self.prevx, self.prevy = 0.0, 0.0

        self.lane_orientation_sub = rospy.Subscriber('/lane_orientation', Float32, self.lane_orientation_callback)
        self.lane_orientation = 0.0

        # # Stop Sign Detector
        # self.ssd = StopSignDetector()


        # read waypoints into the system 
        self.goal       = 0            
        self.read_waypoints() 

        self.desired_speed = 1.5  # m/s, reference speed
        self.max_accel     = 0.48 # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

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

    def inspva_callback(self, inspva_msg):
        if self.latflag == False and self.lonflag == False:
            self.olat = inspva_msg.latitude
            self.olon = inspva_msg.longitude
            self.latflag, self.lonflag = True, True
        if self.latflag == True and self.lonflag == True:
            self.lat     = inspva_msg.latitude  # latitude
            self.lon     = inspva_msg.longitude # longitude
            self.lon_wp_x, self.lat_wp_y = axy.ll2xy(self.lat, self.lon, self.olat, self.olon)
        # self.lon_wp_x = self.lon_wp_x - 0.46 * np.cos(self.ekf_heading)
        # self.lat_wp_y = self.lat_wp_y - 0.46 * np.sin(self.ekf_heading)
        

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

    def lane_orientation_callback(self, msg):
        self.lane_orientation = msg.data
        print(self.lane_orientation)
        # steering_angle = self.lateral_PID_controller(self.lane_orientation)
        # print("steering angle ", steering_angle)
    
    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    def enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle

    def read_waypoints(self):
        # read recorded GPS lat, lon, heading
        dirname  = os.path.dirname(__file__)

        if not GEM_CAR:
            filename = os.path.join(dirname, '/home/paulosuma/Documents/SafeAuto/mp-release-23fa-main/src/gem_Project/output_odom_data.csv')
        else:
            filename = os.path.join(dirname, '/home/ece484/catkin_ws/src/vehicle_control/center_lane_origin.csv')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f)]
        # x towards East and y towards North
        self.path_points_lon_x   = [float(point[0]) for point in path_points] 
        self.path_points_lat_y   = [float(point[1]) for point in path_points] 
        self.path_points_heading = [float(point[2]) for point in path_points] # heading global
        self.wp_size             = len(self.path_points_lon_x)
        self.dist_arr            = np.zeros(self.wp_size)

    def get_gem_state(self):
        
        local_x_curr, local_y_curr = self.ekf_x, self.ekf_y
        curr_yaw = self.ekf_heading

        # reference point is located at the center of rear axle
        curr_x = local_x_curr - self.offset * np.cos(curr_yaw)
        curr_y = local_y_curr - self.offset * np.sin(curr_yaw)

        #Rtab_update
        samex = (curr_x == self.prevx)
        samey = (curr_y == self.prevy)

        if samex and samey and curr_x != 0.0 and curr_y != 0.0:
            pass
        else:
            self.prevx, self.prevy = curr_x, curr_y
            print("rtab", round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4))
            print("gps", self.lon_wp_x, self.lat_wp_y)
        print("c", self.lon_wp_x, self.lat_wp_y)
        return round(curr_x, 3), round(curr_y, 3), round(curr_yaw, 4)

    # find the angle bewtween two vectors    
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        # [-pi, pi]
        return np.arctan2(sinang, cosang)

    # computes the Euclidean distance between two 2D points
    def dist(self, p1, p2):
        return round(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 3)

    def start_pp(self):
            
            if(self.pacmod_enable == True):

                if (self.gem_enable == False):

                    # ---------- enable PACMod ----------

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

                    # self.gear_pub.publish(self.gear_cmd)
                    # print("Foward Engaged!")

                    # self.turn_pub.publish(self.turn_cmd)
                    # print("Turn Signal Ready!")
                    
                    # self.brake_pub.publish(self.brake_cmd)
                    # print("Brake Engaged!")

                    # self.accel_pub.publish(self.accel_cmd)
                    # print("Gas Engaged!")

                    # self.gem_enable = True


            self.path_points_x = np.array(self.path_points_lon_x)
            self.path_points_y = np.array(self.path_points_lat_y)

            curr_x, curr_y, curr_yaw = self.get_gem_state()

            # finding the distance of each way point from the current position
            for i in range(len(self.path_points_x)):
                self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (curr_x, curr_y))

            # finding those points which are less than the look ahead distance (will be behind and ahead of the vehicle)
            goal_arr = np.where( (self.dist_arr < self.look_ahead + 0.3) & (self.dist_arr > self.look_ahead - 0.3) )[0]
            # print("goal array", goal_arr)
            # finding the goal point which is the last in the set of points less than the lookahead distance
            # for idx in goal_arr:
            #     v1 = [self.path_points_x[idx]-curr_x , self.path_points_y[idx]-curr_y]
            #     v2 = [np.cos(curr_yaw), np.sin(curr_yaw)]
            #     temp_angle = self.find_angle(v1,v2)
            #     # find correct look-ahead point by using heading information
            #     if abs(temp_angle) < np.pi:
            #         self.goal = idx
            #         break

            for idx in goal_arr:
                P_T_G = np.array([self.path_points_x[idx], self.path_points_y[idx], 1])
                # print("target x, y = ", self.path_points_x[idx], self.path_points_y[idx])
                Homo_B_G = np.array([[math.cos(curr_yaw), -math.sin(curr_yaw), curr_x], 
                                    [math.sin(curr_yaw), math.cos(curr_yaw), curr_y], 
                                    [0, 0, 1]])
                P_T_B = np.matmul(np.linalg.inv(Homo_B_G), P_T_G)
                if P_T_B[0] > 0: 
                    self.goal = idx
                    break


            # finding the distance between the goal point and the vehicle
            # true look-ahead distance between a waypoint and current position
            L = self.dist_arr[self.goal]

            # find the curvature and the angle 
            P_T_G = np.array([self.path_points_x[self.goal], self.path_points_y[self.goal], 1])
            # print("target x, y = ", self.path_points_x[self.goal], self.path_points_y[self.goal])
            Homo_B_G = np.array([[math.cos(curr_yaw), -math.sin(curr_yaw), curr_x], 
                                [math.sin(curr_yaw), math.cos(curr_yaw), curr_y], 
                                [0, 0, 1]])
            P_T_B = np.matmul(np.linalg.inv(Homo_B_G), P_T_G)

            alpha = math.atan2(P_T_B[1], P_T_B[0])

            # ----------------- tuning this part as needed -----------------
            k       = 0.41 
            angle_i = math.atan((k * 2 * self.wheelbase * math.sin(alpha)) / L) 
            angle   = angle_i*2
            # ----------------- tuning this part as needed -----------------

            f_delta = round(np.clip(angle, -0.61, 0.61), 3)

            f_delta_deg = np.degrees(f_delta)

            # steering_angle in degrees
            steering_angle = self.front2steer(f_delta_deg)
            
            # #------------------------student_vision.py------------------------------#
            # steering_angle = self.front2steer(np.degrees(self.lane_orientation))
            # print("steering angle ", steering_angle)

            if(self.gem_enable == True):
                # print("Current index: " + str(self.goal))
                # print("Forward velocity: " + str(self.speed))
                ct_error = round(np.sin(alpha) * L, 3)
                # print("Crosstrack Error: " + str(ct_error))
                # print("Front steering angle: " + str(np.degrees(f_delta)) + " degrees")
                # print("Steering wheel angle: " + str(steering_angle) + " degrees" )
                # print("\n")

            current_time = rospy.get_time()
            filt_vel     = self.speed_filter.get_data(self.speed)
            output_accel = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)

            if output_accel > self.max_accel:
                output_accel = self.max_accel

            if output_accel < 0.3:
                output_accel = 0.3

            if (f_delta_deg <= 30 and f_delta_deg >= -30):
                self.turn_cmd.ui16_cmd = 1
            elif(f_delta_deg > 30):
                self.turn_cmd.ui16_cmd = 2 # turn left
            else:
                self.turn_cmd.ui16_cmd = 0 # turn right


            # ######### Stop sign ###########
            # if self.ssd.detect_stop_sign():
            #     self.accel_pub.publish(-1)
            # else:
            #     self.accel_pub.publish(self.accel_cmd)


            self.accel_cmd.f64_cmd = output_accel
            self.steer_cmd.angular_position = np.radians(steering_angle)
            self.accel_pub.publish(self.accel_cmd)
            self.steer_pub.publish(self.steer_cmd)
            self.turn_pub.publish(self.turn_cmd)

            self.rate.sleep()


def pure_pursuit():

    rospy.init_node('gnss_pp_node', anonymous=True)
    pp = PurePursuit()

    while not rospy.core.is_shutdown():
        try:
            pp.start_pp()
            # pp.get_gem_state()
        except rospy.ROSInterruptException:
            pass


if __name__ == '__main__':
    pure_pursuit()


