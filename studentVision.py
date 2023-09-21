import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology
import matplotlib.pyplot as plt

# import rosbag
# import os



class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        #self.sub_image = rospy.Subscriber('/gem/front_single_camera/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        mask_image, bird_image = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)


    def gradient_thresh(self, img, thresh_min=75, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to uint8, then apply threshold to get binary image
        ## TODO
        scale = 1
        delta = 0
        ddepth = cv2.CV_8U

        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(grayscale, (3,3), 0)

        x_grad = cv2.Sobel(blurred_img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        y_grad = cv2.Sobel(blurred_img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        combinedgrad = cv2.addWeighted(x_grad, 0.5, y_grad, 0.5, 0)
        range_thresh = cv2.inRange(combinedgrad, thresh_min, thresh_max)

        #max_thresholding
        thres, binary_output = cv2.threshold(range_thresh, thresh_min, 255, cv2.THRESH_BINARY)
        
        # plt.imshow(binary_output)
        # plt.show()

        ####

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls_img)
        range_thresh = cv2.inRange(s, thresh[0], thresh[1])
        thres, binary_output = cv2.threshold(range_thresh, thresh[0], 255, cv2.THRESH_BINARY)

        ####
        # fig = plt.figure()
        # r, c = 1, 2
        # fig.add_subplot(r, c, 1)
        # plt.imshow(img)
        # fig.add_subplot(r, c, 2)
        # plt.imshow(binary_output)
        # plt.show()

        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO

        ####

        binaryImage = np.zeros_like(self.gradient_thresh(img))
        binaryImage[(self.color_thresh(img)==255)|(self.gradient_thresh(img)==255)] = 255

        # spec_image = self.color_thresh(img)
        # # print(np.where(spec_image[:, 0] == 255))
        # # print(np.where(spec_image[:, 639] == 255))
        # maxzerocol = 0
        # c = 0
        # for i in range(640):
        #     count = np.count_nonzero(spec_image[:, i] == 0)
        #     if count > c:
        #         maxzerocol = i
        #         c =count
        # print(maxzerocol)
        # print(np.where(spec_image[:, 311] == 0 ))

        # #GAZEBO MASK
        # points = np.array([[(0, 390), (0,480), (640,480), (640, 307), (321, 239)]])

        #RVIZ MASK
        points = np.array([[(220, 375), (650, 170), (790,375)]])
        pnts_in_lane = np.array([[(425,375), (720,375), (600,210), (650,210)]])

        mask = np.zeros_like(img)
        graymask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.fillPoly(graymask, points, color=(255, 255, 255))
        cv2.fillPoly(graymask, pnts_in_lane, color=(0, 0, 0))
        binaryImage = cv2.bitwise_and(binaryImage, graymask, mask=None)

        # Remove noise from binary image
        #binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)
        # fig = plt.figure()
        # r, c = 1, 2
        # fig.add_subplot(r, c, 1)
        # plt.imshow(img)
        # fig.add_subplot(r, c, 2)
        # plt.imshow(binaryImage)
        # plt.show()

        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO
        #source points from eyeball
        # frame = cv2.resize(img, (640, 480))

        #GAZEBO
        # tl = (275, 255)
        # bl = (5, 400)
        # tr = (365, 255)
        # br = (640, 400)
        # cv2.circle(frame, tl, 5, (0,0,255), -1)
        # cv2.circle(frame, tr, 5, (0,0,255), -1)
        # cv2.circle(frame, bl, 5, (0,0,255), -1)
        # cv2.circle(frame, br, 5, (0,0,255), -1)
        # source_pnts = np.array([tl, bl, tr, br])
        # dest_pnts = np.array([[0,0], [0,480], [640,0], [640,480]])


        #RVIZ
        tl = (560, 200)
        bl = (350, 350)
        tr = (690, 200)
        br = (825, 350)
        source_pnts = np.array([tl, bl, tr, br])
        dest_pnts = np.array([[0,0], [0,480], [640,0], [640,480]])



        source_pnts = np.float32(source_pnts[:, np.newaxis, :])
        dest_pnts = np.float32(dest_pnts[:, np.newaxis, :])

        M = cv2.getPerspectiveTransform(source_pnts, dest_pnts)
        Minv = np.linalg.inv(M)

        warped_img = cv2.warpPerspective(img, M, (640,480))


        ####
        # fig = plt.figure()
        # r, c = 1, 2
        # fig.add_subplot(r, c, 1)
        # plt.imshow(img)
        # fig.add_subplot(r, c, 2)
        # plt.imshow(warped_img)
        # plt.show()


        return warped_img, M, Minv


    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                ret = tune_fit(img_birdeye, left_fit, right_fit)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img

    # def export_image(self):
        
    #     bag = rosbag.Bag("./src/mp1/bags/0484_sync.bag", "r")
    #     bridge = CvBridge()
    #     count = 0
    #     for topic, msg, t in bag.read_messages('camera/image_raw'):
    #         cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    #         cv2.imwrite(os.path.join("./src/image_view/0484", "frame%06i.png" % count), cv_img)
    #         print ("Wrote image %i" % count)

    #         count += 1

    #     bag.close()





if __name__ == '__main__':
    # init args
    # rospy.init_node('lanenet_node', anonymous=True)
    # lanenet_detector()

    # while not rospy.core.is_shutdown():
    #     rospy.rostime.wallsleep(0.5)

    path = "./src/mp1/src/0056.png"
    img = cv2.imread(path)
    ld = lanenet_detector()
    gradient_image = ld.gradient_thresh(img)
    color_image = ld.color_thresh(img)
    combined_image = ld.combinedBinaryImage(img)
    warped_img, M, Minv = ld.perspective_transform(combined_image)
    line_fit(warped_img)





