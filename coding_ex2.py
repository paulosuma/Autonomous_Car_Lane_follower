# Student name: 

import math
import numpy as np
from numpy import linalg as LA
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Accel
from tf2_ros import TransformBroadcaster

from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
import time
from mobile_robotics.utils import quaternion_from_euler, lonlat2xyz, quat2euler


class ExtendedKalmanFilter(Node):

    
    def __init__(self):
        super().__init__('ExtendedKalmanFilter')
        
        
        #array to save the sensor measurements from the rosbag file
        #measure = [p, q, r, fx, fy, fz, x, y, z, vx, vy, vz] 
        self.measure = np.zeros(12)
        
        #Initialization of the variables used to generate the plots.
        self.PHI = []  
        self.PSI = []
        self.THETA = []
        self.P_R = []
        self.P_R1 = []
        self.P_R2 = []
        self.Pos = []
        self.Vel = []
        self.Quater = []
        self.measure_PosX = []
        self.measure_PosY = []
        self.measure_PosZ = []
        self.P_angular = []
        self.Q_angular = []
        self.R_angular = []
        self.P_raw_angular = []
        self.Q_raw_angular = []
        self.R_raw_angular = []
        self.Bias =[]
        
        self.POS_X = []
        self.POS_Y = []
        
        
        #Initialization of the variables used in the EKF
        
        # initial bias values, these are gyroscope and accelerometer biases
        self.bp= 0.0
        self.bq= 0.0
        self.br= 0.0
        self.bfx = 0.0
        self.bfy = 0.0
        self.bfz = 0.0
        # initial rotation
        self.q2, self.q3, self.q4, self.q1 = quaternion_from_euler(0.0, 0.0, np.pi/2) #[qx,qy,qz,qw]

        #initialize the state vector [x y z vx vy vz          quat    gyro-bias accl-bias]
        self.xhat = np.array([[0, 0, 0, 0, 0, 0, self.q1, self.q2, self.q3, self.q4, self.bp, self.bq, self.br, self.bfx, self.bfy, self.bfz]]).T

        self.rgps=np.array([-0.15, 0 ,0]) #This is the location of the GPS wrt CG, this is very important
        # self.rgps=np.array([0, 0 ,0]) 

        #noise params process noise (my gift to you :))
        self.Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.5, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        # self.Q = 100 * self.Q
        #measurement noise
        #GPS position and velocity
        self.R = np.diag([10, 10, 10, 2, 2, 2])
        # self.R = 100*self.R
        
       
        #Initialize P, the covariance matrix
        self.P = np.diag([30, 30, 30, 3, 3, 3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.Pdot=self.P*0.0
        
        self.time = []
        self.loop_t = 0

        # You might find these blocks useful when assembling the transition matrices
        self.Z = np.zeros((3,3))
        self.I = np.eye(3,3)
        self.Z34 = np.zeros((3,4))
        self.Z43 = np.zeros((4,3))
        self.Z36 = np.zeros((3,6))

        self.lat = 0
        self.lon = 0
        self.lat0 = 0
        self.lon0 = 0
        self.flag_lat = False
        self.flag_lon = False
        self.callbacklat_flag = False
        self.callbacklon_flag = False
        self.dt = 0.0125 #set sample time

        # Ros subscribers and publishers
        self.subscription_imu = self.create_subscription(Imu, 'terrasentia/imu', self.callback_imu, 10)
        self.subscription_gps_lat = self.create_subscription(Float32, 'gps_latitude', self.callback_gps_lat, 10)
        self.subscription_gps_lon = self.create_subscription(Float32, 'gps_longitude', self.callback_gps_lon, 10)
        self.subscription_gps_speed_north = self.create_subscription(Float32, 'gps_speed_north', self.callback_gps_speed_north, 10)
        self.subscription_gps_speend_east = self.create_subscription(Float32, 'gps_speed_east', self.callback_gps_speed_east, 10)
        
        self.timer_ekf = self.create_timer(self.dt, self.ekf_callback)
        self.timer_plot = self.create_timer(1, self.plot_data_callback)

        #odometry publisher 
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10) #keep in mind how to declare publishers for next assignments
        self.timer = self.create_timer(0.1, self.timer_callback_odom) #It creates a timer to periodically publish the odometry.

        self.tf_broadcaster = TransformBroadcaster(self) # To broadcast the transformation between coordinate frames.

    def timer_callback_odom(self):
        '''
        Retrieve the estimated position and orientation of the robot
        '''
    
        # We need to set an odometry message and publish the transformation between two coordinate frames
        # Further info about odometry message: https://docs.ros2.org/foxy/api/nav_msgs/msg/Odometry.html
        # Further info about tf2: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html
        # Further info about coordinate frames in ROS: https://www.ros.org/reps/rep-0105.html

        frame_id = 'odom'
        child_frame_id = 'base_link'

        position = [self.xhat[0,0], self.xhat[1,0], self.xhat[2,0]]
        quater = [self.xhat[6,0], self.xhat[7,0], self.xhat[8,0], self.xhat[9,0]]
        
        self.broadcast_tf(position, quater, frame_id, child_frame_id)  # Before creating the odometry message, go to the broadcast_tf function and complete it.
        
        odom = Odometry()
        odom.header.frame_id = frame_id
        odom.header.stamp = self.get_clock().now().to_msg()

        # set the pose. Uncomment next lines

        odom.pose.pose.position.x = self.xhat[0,0] # ...
        odom.pose.pose.position.y = self.xhat[1,0] # ...
        odom.pose.pose.position.z = self.xhat[3,0] # ... 
        odom.pose.pose.orientation.x = self.xhat[6,0]
        odom.pose.pose.orientation.y = self.xhat[7,0] # ...
        odom.pose.pose.orientation.z = self.xhat[8,0] # ...
        odom.pose.pose.orientation.w = self.xhat[9,0] # ...

        # set the velocities. Uncomment next lines
        odom.child_frame_id = child_frame_id
        odom.twist.twist.linear.x = self.xhat[4,0] # ...
        odom.twist.twist.linear.y = self.xhat[5,0] # ...
        odom.twist.twist.linear.z = self.xhat[6,0] # ...

        self.odom_pub.publish(odom)

    def broadcast_tf(self, pos, quater, frame_id, child_frame_id):
        '''
        It continuously publishes the transformation between two reference frames.
        Complete the translation and the rotation of this transformation
        '''
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id

        # Uncomment next lines and complete the code
        t.transform.translation.x = pos[0] # ...
        t.transform.translation.y = pos[1] # ...
        t.transform.translation.z = pos[2] # ...

        t.transform.rotation.x = quater[0] # ...
        t.transform.rotation.y = quater[1] # ...
        t.transform.rotation.z = quater[2] # ...
        t.transform.rotation.w = quater[3] # ...

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    
    def callback_imu(self,msg):
        #measurement vector = [p, q, r, fx, fy, fz, x, y, z, vx, vy, vz]
        # In practice, the IMU measurements should be filtered. In this coding exercise, we are just going to clip
        # the values of velocity and acceleration to keep them in physically possible intervals.
        self.measure[0] = np.clip(msg.angular_velocity.x,-5,5) #(-5,5)
        self.measure[1] = np.clip(msg.angular_velocity.y,-5,5) #(-5,5)
        self.measure[2] = np.clip(msg.angular_velocity.z,-5,5) #(-5,5)
        self.measure[3] = np.clip(msg.linear_acceleration.x,-6,6) #..(-6,6)
        self.measure[4] = np.clip(msg.linear_acceleration.y,-6,6) #..(-6,6)
        self.measure[5] = np.clip(msg.linear_acceleration.z,-16,4) #..(-16,-4)

 
    def callback_gps_lat(self, msg):
        self.lat = msg.data
        if (self.flag_lat == False): #just a trick to recover the initial value of latitude
            self.lat0 = msg.data
            self.flag_lat = True
        
        if (self.flag_lat and self.flag_lon): 
            x, y = lonlat2xyz(self.lat, self.lon, self.lat0, self.lon0) # convert latitude and longitude to x and y coordinates
            self.measure[6] = x
            self.measure[7] = y
            self.measure[8] = 0.0 
        self.callbacklat_flag = True

    
    def callback_gps_lon(self, msg):
        self.lon = msg.data
        if (self.flag_lon == False): #just a trick to recover the initial value of longitude
            self.lon0 = msg.data
            self.flag_lon = True 
        self.callbacklon_flag = True   
    
    def callback_gps_speed_east(self, msg): 
        self.measure[9] = msg.data # vx
        self.measure[11] = 0.0 # vz

    def callback_gps_speed_north(self, msg):
        self.measure[10] = msg.data # vy

   
    def ekf_callback(self):
        #print("iteration:  ",self.loop_t)
        if (self.flag_lat and self.flag_lon):  #Trick  to sincronize rosbag with EKF
            self.ekf_function()
        else:
            print("Play the rosbag file...")

    
    
    def ekf_function(self):
        
        # Adjusting angular velocities and acceleration with the corresponding bias
        self.p = self.measure[0]-self.xhat[10,0] #p-bp angular rates from imu + bias terms
        self.q = self.measure[1]-self.xhat[11,0]
        self.r = self.measure[2]-self.xhat[12,0]
        self.fx = self.measure[3]-self.xhat[13,0]     #linear acceleration from imu + bias terms
        self.fy = self.measure[4]-self.xhat[14,0]
        self.fz = self.measure[5]-self.xhat[15,0]
        
        # Get the current quaternion values from the state vector
        # Remember again the state vector [x y z vx vy vz q1 q2 q3 q4 bp bq br bx by bz]
        self.quat = np.array([[self.xhat[6,0], self.xhat[7,0], self.xhat[8,0], self.xhat[9,0]]]).T
    
        self.q1 = self.quat[0,0]
        self.q2 = self.quat[1,0]
        self.q3 = self.quat[2,0]
        self.q4 = self.quat[3,0]
                
        # Rotation matrix: body to inertial frame
        self.R_bi = np.array([[pow(self.q1,2)+pow(self.q2,2)-pow(self.q3,2)-pow(self.q4,2), 2*(self.q2*self.q3-self.q1*self.q4), 2*(self.q2*self.q4+self.q1*self.q3)],
                          [2*(self.q2*self.q3+self.q1*self.q4), pow(self.q1,2)-pow(self.q2,2)+pow(self.q3,2)-pow(self.q4,2), 2*(self.q3*self.q4-self.q1*self.q2)],
                          [2*(self.q2*self.q4-self.q1*self.q3), 2*(self.q3*self.q4+self.q1*self.q2), pow(self.q1,2)-pow(self.q2,2)-pow(self.q3,2)+pow(self.q4,2)]])
        
            
        #Prediction step
        #First write out all the dots for all the states, e.g. pxdot, pydot, q1dot etc
        #[x y z vx vy vz          quat    gyro-bias accl-bias]
        #[x y z vx vy vz q1 q2 q3 q4 bp bq br bx by bz]
        pxdot = self.xhat[3,0] #vx
        pydot = self.xhat[4,0]
        pzdot = self.xhat[5,0]

        v_i_dot = np.matmul(self.R_bi, np.array([self.fx, self.fy, self.fz]).T)
        vxdot = v_i_dot[0]
        vydot = v_i_dot[1]
        vzdot = v_i_dot[2]

        Omega = np.array([[0, self.p, self.q, self.r], 
                          [-self.p, 0, -self.r, self.q],
                          [-self.q, self.r, 0, -self.p],
                          [-self.r, -self.q, self.p, 0]])
        quat_dot = -0.5*np.matmul(Omega, self.quat)
        q1dot = quat_dot[0] 
        q2dot = quat_dot[1] 
        q3dot = quat_dot[2] 
        q4dot = quat_dot[3] 

        # .. your code here
        
        #Now integrate Euler Integration for Process Updates and Covariance Updates
        # Euler works fine
        # Remember again the state vector [x y z vx vy vz q1 q2 q3 q4 bp bq br bx by bz]
        self.xhat[0,0] = self.xhat[0,0] + self.dt*pxdot
        self.xhat[1,0] = self.xhat[1,0] + self.dt*pydot
        self.xhat[2,0] = self.xhat[2,0] + self.dt*pydot
        self.xhat[3,0] = self.xhat[3,0] + self.dt*vxdot
        self.xhat[4,0] = self.xhat[4,0] + self.dt*vydot
        self.xhat[5,0] = self.xhat[5,0] + self.dt*(vzdot+9.801) #Do not forget Gravity (9.801 m/s2) 
        self.xhat[6,0] = self.xhat[6,0] + self.dt*q1dot
        self.xhat[7,0] = self.xhat[7,0] + self.dt*q2dot
        self.xhat[8,0] = self.xhat[8,0] + self.dt*q3dot
        self.xhat[9,0] = self.xhat[9,0] + self.dt*q4dot

        print("x ekf: ", self.xhat[0,0])
        print("y ekf: ", self.xhat[1,0])
        print("z ekf: ", self.xhat[2,0])
        
        # Extract and normalize the quat    
        self.quat = np.array([[self.xhat[6,0], self.xhat[7,0], self.xhat[8,0], self.xhat[9,0]]]).T
        # .. Normailize quat
        q_magnitude = np.linalg.norm(self.quat)
        self.quat = self.quat/q_magnitude # code here. Uncomment this line
        
        #re-assign quat
        self.xhat[6,0] = self.quat[0,0]
        self.xhat[7,0] = self.quat[1,0]
        self.xhat[8,0] = self.quat[2,0]
        self.xhat[9,0] = self.quat[3,0]
        
                
        # Now write out all the partials to compute the transition matrix Phi
        #delV/delQ
        q_1, q_2, q_3, q_4 = self.quat[0,0], self.quat[1,0], self.quat[2,0], self.quat[3,0]
        a_x, a_y, a_z = self.fx, self.fy, self.fz
        Fvq = np.array([[2*(q_1*a_x+q_4*a_y-q_3*a_z), 2*(q_2*a_x+q_3*a_y+q_4*a_z), 2*(-q_3*a_x+q_2*a_y+q_1*a_z), 2*(-q_4*a_x-q_1*a_y+q_2*a_z)], 
                        [2*(q_4*a_x+q_1*a_y-q_2*a_z), 2*(q_3*a_x-q_2*a_y-q_1*a_z), 2*(q_2*a_x+q_3*a_y+q_4*a_z), 2*(q_1*a_x-q_4*a_y+q_3*a_z)], 
                        [2*(-q_3*a_x+q_2*a_y+q_1*a_z), 2*(q_4*a_x+q_1*a_y-q_2*a_z), 2*(-q_1*a_x+q_4*a_y-q_3*a_z), 2*(q_2*a_x+q_3*a_y+q_4*a_z)]])
        #delV/del_abias
        
        Fvb = -self.R_bi
        
        #delQ/delQ
        
        Fqq = -0.5*Omega
     
        #delQ/del_gyrobias
        Fqb = 0.5*np.array([[q_2, q_3, q_4],
                            [-q_1, q_4, -q_3],
                            [-q_4, -q_1, q_2],
                            [q_3, -q_2, -q_1]])
        # Now assemble the Transition matrix A
        r1 = np.hstack((self.Z, self.I, self.Z34, self.Z, self.Z)) 
        r2 = np.hstack((self.Z, self.Z, Fvq, self.Z, Fvb))
        r3 = np.hstack((self.Z43, self.Z43, Fqq, Fqb, self.Z43))
        r4 = np.hstack((self.Z, self.Z, self.Z34, self.Z, self.Z))
        r5 = np.hstack((self.Z, self.Z, self.Z34, self.Z, self.Z))
        
        A = np.vstack((r1, r2, r3, r4, r5))
        
        #Propagate the error covariance matrix, I suggest using the continuous integration since Q, R are not discretized 
        #Pdot = A@P+P@A.transpose() + Q
        #P = P +Pdot*dt
        self.Pdot = np.matmul(A, self.P) + np.matmul(self.P, A.T) + self.Q
        self.P = self.P + self.Pdot*self.dt
        
        #Correction step
        #Get measurements 3 positions and 3 velocities from GPS
        self.z = np.array([[self.measure[6], self.measure[7], self.measure[8], self.measure[9], self.measure[10], self.measure[11]]]).T #x y z vx vy vz
        #Write out the measurement matrix linearization to get H
        
        # del v/del q
        rg = self.rgps[0]

        Hvq = 2*np.array([[rg*q_3*self.q+rg*q_4*self.r, rg*q_4*self.q-rg*q_3*self.r, rg*q_1*self.q-rg*q_2*self.r, rg*q_2*self.q+rg*q_1*self.r], 
                          [-rg*q_2*self.q-rg*q_1*self.r, rg*q_2*self.r-rg*q_1*self.q, rg*q_4*self.q-rg*q_3*self.r, rg*q_3*self.q+rg*q_4*self.r], 
                          [rg*q_1*self.q-rg*q_2*self.r, -rg*q_2*self.q-rg*q_1*self.r, -rg*q_3*self.q-rg*q_4*self.r, rg*q_4*self.q-rg*q_3*self.r]])
        
        #del P/del q
        Hxq = 2*np.array([[-rg*q_1, -rg*q_2, rg*q_3, rg*q_4],
                          [-rg*q_4, -rg*q_3, -rg*q_2, -rg*q_1],
                          [rg*q_3, -rg*q_4, rg*q_1, -rg*q_2]])
        
        # Assemble H
        h_1 = np.hstack((self.I, self.Z, Hxq, self.Z36))
        h_2 = np.hstack((self.Z, self.I, Hvq, self.Z36))
        H = np.vstack((h_1, h_2))

        #Compute Kalman gain
        coeff = np.linalg.inv(np.matmul(np.matmul(H,self.P), H.T) + self.R)
        L = np.matmul(np.matmul(self.P, H.T), coeff)
        if self.callbacklat_flag and self.callbacklon_flag:
            #Perform xhat correction    xhat = xhat + L@(z-H@xhat)
            self.xhat = self.xhat +  np.matmul(L, (self.z - np.matmul(H, self.xhat)))
            
            #propagate error covariance approximation P = (np.eye(16,16)-L@H)@P
            self.P = np.matmul((np.eye(16,16) - np.matmul(L,H)), self.P)

            self.callbacklat_flag, self.callbacklon_flag = False, False

        #Now let us do some book-keeping 
        # Get some Euler angles
        
        phi, theta, psi = quat2euler(self.quat.T)

        self.PHI.append(phi*180/math.pi)
        self.THETA.append(theta*180/math.pi)
        self.PSI.append(psi*180/math.pi)
    
          
        # Saving data for the plots. Uncomment the 4 lines below once you have finished the ekf function

        DP = np.diag(self.P)
        self.P_R.append(DP[0:3])
        self.P_R1.append(DP[3:6])
        self.P_R2.append(DP[6:10])
        self.Pos.append(self.xhat[0:3].T[0])
        self.POS_X.append(self.xhat[0,0])
        self.POS_Y.append(self.xhat[1,0])
        self.Vel.append(self.xhat[3:6].T[0])
        self.Quater.append(self.xhat[6:10].T[0])
        self.Bias.append(self.xhat[10:16].T[0])
        B = self.measure[6:9].T
        self.measure_PosX.append(B[0])
        self.measure_PosY.append(B[1])
        self.measure_PosZ.append(B[2])

        self.P_angular.append(self.p)
        self.Q_angular.append(self.q)
        self.R_angular.append(self.r)

        self.loop_t += 1
        self.time.append(self.loop_t*self.dt)

    def plot_data_callback(self):

        plt.figure(1)
        plt.clf()
        plt.plot(self.time,self.PHI,'b', self.time, self.THETA, 'g', self.time,self.PSI, 'r')
        plt.legend(['phi','theta','psi'])
        plt.title('Phi, Theta, Psi [deg]')

        plt.figure(2)
        plt.clf()
        plt.plot(self.measure_PosX, self.measure_PosY, self.POS_X, self.POS_Y)
        plt.title('xy trajectory')
        plt.legend(['GPS','EKF'])

        plt.figure(3)
        plt.clf()
        plt.plot(self.time,self.P_R)
        plt.title('Covariance of Position')
        plt.legend(['px','py','pz'])
        # plt.figure(4)
        # plt.clf()
        # plt.plot(self.time,self.P_R1)
        # plt.legend(['pxdot','pydot','pzdot'])
        # plt.title('Covariance of Velocities')
        # plt.figure(5)
        # plt.clf()
        # plt.plot(self.time,self.P_R2)
        # plt.title('Covariance of Quaternions')
        # plt.figure(6)
        # plt.clf()
        # plt.plot(self.time,self.Pos,self.time,self.measure_PosX,'r:', self.time,self.measure_PosY,'r:', self.time,self.measure_PosZ,'r:')
        # plt.legend(['X_ekf', 'Y_ekf', 'Z_ekf','Xgps','Ygps','Z_0'])
        # plt.title('Position')
        # plt.figure(7)
        # plt.clf()
        # plt.plot(self.time,self.Vel)
        # plt.title('vel x y z')
        # plt.figure(8)
        # plt.clf()
        # plt.plot(self.time,self.Quater)
        # plt.title('Quat')
        # plt.figure(9)
        # plt.clf()
        # plt.plot(self.time,self.P_angular,self.time,self.Q_angular,self.time,self.R_angular)
        # plt.title('OMEGA with Bias')
        # plt.legend(['p','q','r'])

        plt.figure(10)
        plt.clf()
        plt.plot(self.time,self.Bias)
        plt.title('Gyroscope and accelerometer Bias')
        plt.legend(['bp','bq','br','bfx','bfy','bfz'])
                
        plt.ion()
        plt.show()
        plt.pause(0.0001)
        

def main(args=None):
    rclpy.init(args=args)

    ekf_node = ExtendedKalmanFilter()

    rclpy.spin(ekf_node)

   
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
