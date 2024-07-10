import time
import math
import numpy as np
import cv2
import rospy
import os

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
import matplotlib.pyplot as plt
from ackermann_msgs.msg import AckermannDrive

import rosbag
import os



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        # self.sub_image = rospy.Subscriber('/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        self.sub_image = rospy.Subscriber("/zed2/zed_node/right/image_rect_color", Image, self.img_callback, queue_size=1)
        # self.sub_image = rospy.Subscriber("/zed2/zed_node/rgb/image_rect_color", Image, self.img_callback, queue_size=1)
        self.pub_overlay = rospy.Publisher("lane_detection/overlay", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        self.pub_lane_orientation = rospy.Publisher('/lane_orientation', Float32, queue_size=1)


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # print(cv_image.shape)
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        lane_overlay, birds_eye = self.detection(raw_img)

        if lane_overlay is not None and birds_eye is not None:
            # Convert an OpenCV image into a ROS image message
            # plt.imshow(mask_image)
            # plt.show()
            out_lane_overlay = self.bridge.cv2_to_imgmsg(lane_overlay, 'bgr8')
            out_birds_eye = self.bridge.cv2_to_imgmsg(birds_eye, 'bgr8')
            # Publish image message in ROS
            self.pub_overlay.publish(out_lane_overlay)
            self.pub_bird.publish(out_birds_eye)


    def detection(self, img):

        orginal_im = img
        gray_im = cv2.cvtColor(orginal_im, cv2.COLOR_BGR2GRAY)
        blur_im = cv2.GaussianBlur(gray_im, (3,3), 0)
        edge_im = cv2.Canny(blur_im, 50, 200, None, 3)
        ret, edge_im = cv2.threshold(edge_im,250,255,cv2.THRESH_BINARY)
        # print(type(edge_im))

    

        #points for Gazebo
        # points = np.array([[(0, 390), (0,465), (640,465), (640, 307), (321, 239)]]) 
        #points for GemCar
        points = np.array([[(300, 660), (550,425), (800,425), (1050, 660)]]) 
        masked_im = np.zeros_like(gray_im)
        cv2.fillPoly(masked_im, points, color=(255, 255, 255))

        combined_image = cv2.bitwise_and(edge_im, masked_im, mask=None)

        # Copy edges to the images that will display the results in BGR
        cdstP = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2BGR)
        # linesP = cv2.HoughLinesP(combined_image, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 150, minLineLength=50, maxLineGap=50)
        linesP = cv2.HoughLinesP(combined_image, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 50, minLineLength=70, maxLineGap=50)
        if linesP is None: print(linesP) 

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
        lane_overlay = cv2.addWeighted(img,1,cdstP,1,1)
        # print(lane_overlay.shape)

        plt.imshow(lane_overlay)
        plt.show()


        #Bird's eye view for Gem
        tl = (500, 425)
        bl = (280, 660)
        tr = (780, 425)
        br = (1000, 660)
        source_pnts = np.float32([tl, bl, tr, br])
        dest_pnts = np.float32([[0,0], [0,720], [1280,0], [1280,720]])
        # for i in range(0,4):
                # cv2.circle(lane_overlay,(source_pnts[i][0], source_pnts[i][1]),5,(0,0,255),2)

        M = cv2.getPerspectiveTransform(source_pnts, dest_pnts)
        birds_eye = cv2.warpPerspective(cdstP, M, (orginal_im.shape[1], orginal_im.shape[0]))
        
        # plt.imshow(birds_eye)
        # plt.show()
        gray_birdseye = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2GRAY)
        gray_birdseye = cv2.GaussianBlur(gray_birdseye, (3,3), 0)
        gray_birdseye = cv2.Canny(gray_birdseye, 50, 95, None, 3)
        # gray_birdseye = cv2.threshold(gray_birdseye,200,255,cv2.THRESH_BINARY)
        # print(gray_birdseye)
        lines_birdseye = cv2.HoughLinesP(gray_birdseye, cv2.HOUGH_PROBABILISTIC, np.pi / 180, 80, minLineLength=70, maxLineGap=50)

        if lines_birdseye is not None:
            curr_angle = 0.0
            n = len(lines_birdseye)
            for l in lines_birdseye:
                x1, y1, x2, y2 = l[0]
                # cv2.line(birds_eye, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #left lane
                if x1 < 640 and x2 < 640 and y1 > 360 and y2 > 360:
                    x1_left = x1
                    x2_left = x2
                    y1_left = y1
                    y2_left = y2
                    cv2.line(birds_eye, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # elif x1 > 640 or x2 > 640 and y1 > 360 and y2 > 360:
                #     x1_right = x1
                #     x2_right = x2
                #     y1_right = y1
                #     y2_right = y2
                
                try:
                    #calculate angle between line and center line
                    if (x1_left > x2_left and y1_left < y2_left) or (x1_left < x2_left and y1_left > y2_left):
                        ang_rad = np.arctan2(abs(x2_left-x1_left), abs(y2_left-y1_left)) 
                        curr_angle+= -ang_rad
                    elif (x1_left > x2_left and y1_left > y2_left) or (x1_left < x2_left and y1_left < y2_left):
                        ang_rad = np.arctan2(abs(x2_left-x1_left), abs(y2_left-y1_left)) 
                        curr_angle+=ang_rad
                except NameError:
                    continue

            curr_angle = curr_angle/n
            # angle = (curr_angle *180 / np.pi)
            # print(curr_angle) 
            #draw center line in the middle of the birds eye 
            xc_1, xc_2 = 640, 640
            yc_1, yc_2 = 720, 420
            cv2.line(birds_eye, (xc_1,yc_1), (xc_2, yc_2), (255, 255, 255), 2)

            x_dist = int(300*math.tan(curr_angle))
            cv2.line(birds_eye, (xc_1,yc_1), (xc_2+x_dist, yc_2), (255, 0, 255), 2)
            if abs(curr_angle) > math.pi/4: 
                print(curr_angle)
                self.pub_lane_orientation.publish(curr_angle)

        return lane_overlay, birds_eye
    

def main():
    im = cv2.imread("./src/gem_Project/gem_pic.png")
    t = lanenet_detector()
    t.detection(im)

main()






# if __name__ == '__main__':
#     # init args
#     rospy.init_node('lanenet_node', anonymous=True)
#     lanenet_detector()
#     print("here")
#     while not rospy.core.is_shutdown():
#         rospy.rostime.wallsleep(0.5)

