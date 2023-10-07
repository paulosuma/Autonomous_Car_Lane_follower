import rospy
from gazebo_msgs.srv import GetModelState, GetModelStateResponse
from gazebo_msgs.msg import ModelState
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from std_msgs.msg import Float32MultiArray
import math
from util import euler_to_quaternion, quaternion_to_euler
import time

class vehicleController():

    def __init__(self):
        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size = 1)
        self.prev_vel = 0
        self.L = 1.75 # Wheelbase, can be get from gem_control.py
        self.log_acceleration = True
        self.prev_alpha = 0

    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp


    # Tasks 1: Read the documentation https://docs.ros.org/en/fuerte/api/gazebo/html/msg/ModelState.html
    #       and extract yaw, velocity, vehicle_position_x, vehicle_position_y
    # Hint: you may use the the helper function(quaternion_to_euler()) we provide to convert from quaternion to euler
    def extract_vehicle_info(self, currentPose):

        ####################### TODO: Your TASK 1 code starts Here #######################
        pos_x, pos_y, vel, yaw = 0, 0, 0, 0
        pos_x = currentPose.pose.position.x
        pos_y = currentPose.pose.position.y

        vel_x = currentPose.twist.linear.x
        vel_y = currentPose.twist.linear.y
        vel = math.sqrt(vel_x**2 + vel_y**2)

        q_x = currentPose.pose.orientation.x
        q_y = currentPose.pose.orientation.y
        q_z = currentPose.pose.orientation.z
        q_w = currentPose.pose.orientation.w
        euler_orientation = quaternion_to_euler(q_x, q_y, q_z, q_w)
        yaw = euler_orientation[2] #radian



        ####################### TODO: Your Task 1 code ends Here #######################

        return pos_x, pos_y, vel, yaw # note that yaw is in radian

    # Task 2: Longtitudal Controller
    # Based on all unreached waypoints, and your current vehicle state, decide your velocity
    def longititudal_controller(self, curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints):

        ####################### TODO: Your TASK 2 code starts Here #######################
        max_velocity = 12

        if len(future_unreached_waypoints) < 5:
            x, y = future_unreached_waypoints[0]
        else:
            x, y = future_unreached_waypoints[3]

        P_T_G = np.array([x, y, 1])
        Homo_B_G = np.array([[math.cos(curr_yaw), -math.sin(curr_yaw), curr_x], 
                          [math.sin(curr_yaw), math.cos(curr_yaw), curr_y], 
                          [0, 0, 1]])
        P_T_B = np.matmul(np.linalg.inv(Homo_B_G), P_T_G)
        alpha = math.atan2(P_T_B[1], P_T_B[0])
        
        # print(alpha)
        if(abs(alpha)>0.3) and (abs(alpha)<0.5):
            return 10.5 * math.cos(alpha)
        if(abs(alpha)>0.5):
            return 9.5 * math.cos(alpha)
        
        target_velocity = max_velocity * math.cos(alpha)

        
        ####################### TODO: Your TASK 2 code ends Here #######################
        return target_velocity


    # Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self, curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints):

        ####################### TODO: Your TASK 3 code starts Here #######################
        target_steering = .8
      
        # if len(future_unreached_waypoints)>=2:
        #     x1, y1 = future_unreached_waypoints[0]
        #     x2, y2 = future_unreached_waypoints[1]
        #     x3, y3 = (x2+x1)/2, (y2+y1)/2
        #     ld = math.sqrt((curr_x-x3)**2 + (curr_y-y3)**2)
        # else:
        # ld = math.sqrt((curr_x-target_point[0])**2 + (curr_y-target_point[1])**2)
        # print(ld)

        max_ld = 10

        P_T_G = np.array([target_point[0], target_point[1], 1])
        Homo_B_G = np.array([[math.cos(curr_yaw), -math.sin(curr_yaw), curr_x], 
                          [math.sin(curr_yaw), math.cos(curr_yaw), curr_y], 
                          [0, 0, 1]])
        P_T_B = np.matmul(np.linalg.inv(Homo_B_G), P_T_G)

        alpha = math.atan2(P_T_B[1], P_T_B[0])

        steering_diff = abs(alpha-self.prev_alpha)
        self.prev_alpha = alpha

        ld = max_ld*math.cos(steering_diff) 

        target_steering = math.atan((2*self.L*math.sin(alpha))/ld)



        ####################### TODO: Your TASK 3 code starts Here #######################
        return target_steering


    def execute(self, currentPose, target_point, future_unreached_waypoints):
        # Compute the control input to the vehicle according to the
        # current and reference pose of the vehicle
        # Input:
        #   currentPose: ModelState, the current state of the vehicle
        #   target_point: [target_x, target_y]
        #   future_unreached_waypoints: a list of future waypoints[[target_x, target_y]]
        # Output: None

        curr_x, curr_y, curr_vel, curr_yaw = self.extract_vehicle_info(currentPose)

        # Acceleration Profile
        if self.log_acceleration:
            acceleration = (curr_vel- self.prev_vel) * 100 # Since we are running in 100Hz

        # print(acceleration)
        self.prev_vel = curr_vel

        print(curr_x, curr_y)

        target_velocity = self.longititudal_controller(curr_x, curr_y, curr_vel, curr_yaw, future_unreached_waypoints)
        target_steering = self.pure_pursuit_lateral_controller(curr_x, curr_y, curr_yaw, target_point, future_unreached_waypoints)


        #Pack computed velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = target_velocity
        newAckermannCmd.steering_angle = target_steering

        # Publish the computed control input to vehicle model
        self.controlPub.publish(newAckermannCmd)

    def stop(self):
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = 0
        self.controlPub.publish(newAckermannCmd)


