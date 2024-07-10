#!/usr/bin/env python3

#================================================================
# File name          : gem_gnss_image.py                                                                  
# Description        : show vehicle's heading and position in an image                                                                
# Author             : Hang Cui (hangcui3@illinois.edu)                                                                     
# Date created       : 08/13/2022                                                                
# Date last modified : 01/25/2023                                                            
# Version            : 0.2                                                                    
# Usage              : rosrun gem_gnss gem_gnss_image.py                                                                      
# Python version     : 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os
import cv2 
import csv
import math
import numpy as np
from numpy import linalg as la

# ROS Headers
import tf
import rospy
import rospkg
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import alvinxy as axy

# GEM Sensor Headers
from std_msgs.msg import Float64
from gps_common.msg import GPSFix
from sensor_msgs.msg import Imu, NavSatFix
from novatel_gps_msgs.msg import NovatelPosition, NovatelXYZ, Inspva, NovatelCorrectedImuData

image_file  = 'gnss_map.png'
curr_path = os.path.abspath(__file__) 
image_path = curr_path.split('scripts')[0] + 'images/' + image_file

class GNSSImage(object):


    def __init__(self):

        self.rate = rospy.Rate(15)

        # Subscribe information from sensors
        self.lat      = 0
        self.lon      = 0
        self.heading  = 0
        self.gnss_sub = rospy.Subscriber("/novatel/inspva", Inspva, self.inspva_callback)
        self.cam_pose_sub = rospy.Subscriber("/zed2/zed_node/odom", Odometry, self.cam_odometry_callback)

        self.lat_start_bt = 40.092722  # 40.09269  
        self.lon_start_l  = -88.236365 # -88.23628
        self.lat_scale    = 0.00062    # 0.0007    
        self.lon_scale    = 0.00136    # 0.00131 

        self.cam_x, self.cam_y, self.cam_z = 0, 0, 0 


    def inspva_callback(self, inspva_msg):
        self.lat     = inspva_msg.latitude
        self.lon     = inspva_msg.longitude
        self.heading = inspva_msg.azimuth 

    def cam_odometry_callback(self, msg):
        self.cam_x = msg.pose.pose.position.x
        self.cam_y = msg.pose.pose.position.y
        self.cam_z = msg.pose.pose.position.z


    def image_heading(self, lon_x, lat_y, heading):
        
        if(heading >=0 and heading < 90):
            angle  = np.radians(90-heading)
            lon_xd = lon_x + int(self.arrow * np.cos(angle))
            lat_yd = lat_y - int(self.arrow * np.sin(angle))

        elif(heading >= 90 and heading < 180):
            angle  = np.radians(heading-90)
            lon_xd = lon_x + int(self.arrow * np.cos(angle))
            lat_yd = lat_y + int(self.arrow * np.sin(angle))  

        elif(heading >= 180 and heading < 270):
            angle = np.radians(270-heading)
            lon_xd = lon_x - int(self.arrow * np.cos(angle))
            lat_yd = lat_y + int(self.arrow * np.sin(angle))

        else:
            angle = np.radians(heading-270)
            lon_xd = lon_x - int(self.arrow * np.cos(angle))
            lat_yd = lat_y - int(self.arrow * np.sin(angle)) 

        return lon_xd, lat_yd  
    

    # def wps_to_local_xy(self, lon_wp, lat_wp):
    #     # convert GNSS waypoints into local fixed frame reprented in x and y
    #     lon_wp_x, lat_wp_y = axy.ll2xy(lat_wp, lon_wp, self.olat, self.olon)
    #     return lon_wp_x, lat_wp_y

    def heading_to_yaw(self, heading_curr):
        if (heading_curr >= 270 and heading_curr < 360):
            yaw_curr = np.radians(450 - heading_curr)
        else:
            yaw_curr = np.radians(90 - heading_curr)
        return yaw_curr
    
    def lonlat2xyz(self, lat, lon, lat0, lon0): 
        # WGS84 ellipsoid constants:
        a = 6378137
        b = 6356752.3142
        e = math.sqrt(1-b**2/a**2)
        x = a*math.cos(math.radians(lat0))*math.radians(lon-lon0)/math.pow(1-e**2*(math.sin(math.radians(lat0)))**2,0.5)
        y = a*(1 - e**2)*math.radians(lat-lat0)/math.pow(1-e**2*(math.sin(math.radians(lat0)))**2,1.5)
        return x, y # x and y coordinates in a reference frame with the origin in lat0, lon0


    def start_gi(self):
        
        while not rospy.is_shutdown():

            # lon_x = int(self.img_width*(self.lon-self.lon_start_l)/self.lon_scale)
            # lat_y = int(self.img_height-self.img_height*(self.lat-self.lat_start_bt)/self.lat_scale)
            # lon_xd, lat_yd = self.image_heading(lon_x, lat_y, self.heading)

            # pub_image = np.copy(self.map_image)
            # cv2.arrowedLine(pub_image, (lon_x, lat_y), (lon_xd, lat_yd), (0, 0, 255), 2)
            # cv2.circle(pub_image, (lon_x, lat_y), 12, (0,0,255), 2)

            
            # x, y = self.lonlat2xyz(self.lat, self.lon, self.lon, self.lon_start_l)
            (x, y) = axy.ll2xy(self.lat, self.lon, self.lat_start_bt, self.lon_start_l)
            theta = self.heading_to_yaw(self.heading)
            # print("x",x)
            # print("y", y)
            # print("theta", theta)
            print(x, y, theta)

            # print(self.cam_x, self.cam_y, self.cam_z)
            
            self.rate.sleep()


def main():

    rospy.init_node('gem_gnss_image_node', anonymous=True)

    gi = GNSSImage()

    try:
    	gi.start_gi()
    except KeyboardInterrupt:
        print ("Shutting down gnss image node.")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

