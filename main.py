########################################################
#  ROS Simulation                                      #
#  @author Gautam Sharma and Yash Mandlik              #
#  Code written for Dr. Spring Berman's MAE 598 course #
########################################################
"""
MIT License

Copyright (c) [2020] [Gautam Sharma and Yash Mandlik]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# !/usr/bin/env python

import rospy
from std_msgs.msg import String
import time
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
from darknet_ros_msgs.msg import *

from matplotlib import style

pos_msg1 = PoseStamped()
pos_msg2 = PoseStamped()
pos_msg3 = PoseStamped()
pos_msg4 = PoseStamped()
pos_msg5 = PoseStamped()
pos_msg6 = PoseStamped()


################################
#         Drone class         #
################################

class Drone:
    """


    """

    def __init__(self):
        rospy.init_node('control', anonymous=True)
        self.pub1 = rospy.Publisher('/uav1/command/pose', PoseStamped, queue_size=10)
        self.pub2 = rospy.Publisher('/uav2/command/pose', PoseStamped, queue_size=10)
        self.pub3 = rospy.Publisher('/uav3/command/pose', PoseStamped, queue_size=10)
        self.pub4 = rospy.Publisher('/uav4/command/pose', PoseStamped, queue_size=10)
        self.pub5 = rospy.Publisher('/uav5/command/pose', PoseStamped, queue_size=10)
        self.pub6 = rospy.Publisher('/uav6/command/pose', PoseStamped, queue_size=10)

        self.data = np.genfromtxt('znew2.csv', delimiter=",", dtype=float)

        self.x1, self.y1 = self.data[:, 0].tolist(), self.data[:, 6].tolist()
        self.x2, self.y2 = self.data[:, 1].tolist(), self.data[:, 7].tolist()
        self.x3, self.y3 = self.data[:, 2].tolist(), self.data[:, 8].tolist()
        self.x4, self.y4 = self.data[:, 3].tolist(), self.data[:, 9].tolist()
        self.x5, self.y5 = self.data[:, 4].tolist(), self.data[:, 10].tolist()
        self.x6, self.y6 = self.data[:, 5].tolist(), self.data[:, 11].tolist()

        self.detect = False

        self.t0 = time.time()

        self.rate = rospy.Rate(100)

    ################################
    #  Object detection callback.  #
    ################################
    def callback(self, data):
        """

        :param data: information received from ROS subscriber
        :return: if person is detected then converge() is called that publishes coordinates of the detected person to
        all the drones.
        """

        for box in data.bounding_boxes:
            a = box.Class
            prob = box.probability
            xmin = 20
            ymin = 21

            if a == 'person' and prob > 0.5:
                self.detect = True
                self.converge(xmin, ymin, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4,
                              self.x5, self.y5, self.x6, self.y6)

    def detection(self):
        """

        :return: subscriber to the bounding boxes given as output by Darknet YOLO \
        (https://github.com/leggedrobotics/darknet_ros), a ROS package developed for object detection
        """

        rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.callback)

    def vel_pub(self):
        """

        :return: Publishes velocity command to the "leader" of the platoon
        """

        pub = rospy.Publisher("/uav1/cmd_vel", Twist, queue_size=10)
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0

        pub.publish(vel_msg)

    def random_walk(self):
        """

        :return: Helps the flock perform random walk using uniform random distribution
        """

        i = 0

        while not rospy.is_shutdown():

            if (self.detect == False):
                rand_1x = np.random.uniform(low=15, high=30, size=())
                rand_1y = np.random.uniform(low=0, high=30, size=())

                rand_2x = np.random.uniform(low=-15, high=15, size=())
                rand_2y = np.random.uniform(low=0, high=30, size=())

                rand_3x = np.random.uniform(low=-30, high=-15, size=())
                rand_3y = np.random.uniform(low=0, high=30, size=())

                rand_4x = np.random.uniform(low=-30, high=-15, size=())
                rand_4y = np.random.uniform(low=-30, high=0, size=())

                rand_5x = np.random.uniform(low=-15, high=15, size=())
                rand_5y = np.random.uniform(low=-30, high=0, size=())

                rand_6x = np.random.uniform(low=15, high=30, size=())
                rand_6y = np.random.uniform(low=-30, high=0, size=())

                pos_msg1.header.frame_id = "uav1/world"
                pos_msg1.pose.position.z = 5
                pos_msg1.pose.position.x = rand_1x
                pos_msg1.pose.position.y = rand_1y
                pos_msg1.pose.orientation.x = 0
                pos_msg1.pose.orientation.y = 0
                pos_msg1.pose.orientation.z = 0
                pos_msg1.pose.orientation.w = 0

                self.pub1.publish(pos_msg1)

                pos_msg2.header.frame_id = "uav2/world"
                pos_msg2.pose.position.z = 5
                pos_msg2.pose.position.x = rand_2x
                pos_msg2.pose.position.y = rand_2y
                pos_msg2.pose.orientation.x = 0
                pos_msg2.pose.orientation.y = 0
                pos_msg2.pose.orientation.z = 0
                pos_msg2.pose.orientation.w = 0

                self.pub2.publish(pos_msg2)

                pos_msg3.header.frame_id = "uav3/world"
                pos_msg3.pose.position.z = 5
                pos_msg3.pose.position.x = rand_3x
                pos_msg3.pose.position.y = rand_3y
                pos_msg3.pose.orientation.x = 0
                pos_msg3.pose.orientation.y = 0
                pos_msg3.pose.orientation.z = 0
                pos_msg3.pose.orientation.w = 0
                self.pub3.publish(pos_msg3)

                pos_msg4.header.frame_id = "uav4/world"
                pos_msg4.pose.position.z = 5
                pos_msg4.pose.position.x = rand_4x
                pos_msg4.pose.position.y = rand_4y
                pos_msg4.pose.orientation.y = 0
                pos_msg4.pose.orientation.z = 0
                pos_msg4.pose.orientation.w = 0
                self.pub4.publish(pos_msg4)

                pos_msg5.header.frame_id = "uav5/world"
                pos_msg5.pose.position.z = 5
                pos_msg5.pose.position.x = rand_5x
                pos_msg5.pose.position.y = rand_5y
                pos_msg5.pose.orientation.x = 0
                pos_msg5.pose.orientation.y = 0
                pos_msg5.pose.orientation.z = 0
                pos_msg5.pose.orientation.w = 0
                self.pub5.publish(pos_msg5)

                pos_msg6.header.frame_id = "uav6/world"
                pos_msg6.pose.position.z = 5
                pos_msg6.pose.position.x = rand_6x
                pos_msg6.pose.position.y = rand_6y
                pos_msg6.pose.orientation.x = 0
                pos_msg6.pose.orientation.y = 0
                pos_msg6.pose.orientation.z = 0
                pos_msg6.pose.orientation.w = 0
                self.pub6.publish(pos_msg6)

                i += 1
                time.sleep(2)

    def converge(self, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6):
        """

        :param x1: x coordinate of UAV 1
        :param y1: y coordinate of UAV 1
        :param x2: x coordinate of UAV 2
        :param y2: y coordinate of UAV 2
        :param x3: x coordinate of UAV 3
        :param y3: y coordinate of UAV 3
        :param x4: x coordinate of UAV 4
        :param y4: y coordinate of UAV 4
        :param x5: x coordinate of UAV 5
        :param y5: y coordinate of UAV 5
        :param x6: x coordinate of UAV 6
        :param y6: y coordinate of UAV 6
        :return: 
        """

        i = 0
        while not rospy.is_shutdown() or i < self.data.shape[0]:

            if (self.detect == True):

                pos_msg1.header.frame_id = "uav1/world"
                pos_msg1.pose.position.z = 5
                pos_msg1.pose.position.x = x1
                pos_msg1.pose.position.y = y1
                pos_msg1.pose.orientation.x = 0
                pos_msg1.pose.orientation.y = 0
                pos_msg1.pose.orientation.z = 0
                pos_msg1.pose.orientation.w = 0

                self.pub1.publish(pos_msg1)

                pos_msg2.header.frame_id = "uav2/world"
                pos_msg2.pose.position.z = 5
                pos_msg2.pose.position.x = x2[i]
                pos_msg2.pose.position.y = y2[i]
                pos_msg2.pose.orientation.x = 5
                pos_msg2.pose.orientation.y = 3
                pos_msg2.pose.orientation.z = 4
                pos_msg2.pose.orientation.w = 0

                self.pub2.publish(pos_msg2)

                pos_msg3.header.frame_id = "uav3/world"
                pos_msg3.pose.position.z = 5
                pos_msg3.pose.position.x = x3[i]
                pos_msg3.pose.position.y = y3[i]
                pos_msg3.pose.orientation.x = 0
                pos_msg3.pose.orientation.y = 0
                pos_msg3.pose.orientation.z = 0
                pos_msg3.pose.orientation.w = 0
                self.pub3.publish(pos_msg3)

                pos_msg4.header.frame_id = "uav4/world"
                pos_msg4.pose.position.z = 5
                pos_msg4.pose.position.x = x4[i]
                pos_msg4.pose.position.y = y4[i]
                pos_msg4.pose.orientation.x = 0
                pos_msg4.pose.orientation.y = 0
                pos_msg4.pose.orientation.z = 0
                pos_msg4.pose.orientation.w = 0
                self.pub4.publish(pos_msg4)

                pos_msg5.header.frame_id = "uav5/world"
                pos_msg5.pose.position.z = 5
                pos_msg5.pose.position.x = x5[i]
                pos_msg5.pose.position.y = y5[i]
                pos_msg5.pose.orientation.x = 0
                pos_msg5.pose.orientation.y = 0
                pos_msg5.pose.orientation.z = 0
                pos_msg5.pose.orientation.w = 0
                self.pub5.publish(pos_msg5)

                pos_msg6.header.frame_id = "uav6/world"
                pos_msg6.pose.position.z = 5
                pos_msg6.pose.position.x = x6[i]
                pos_msg6.pose.position.y = y6[i]
                pos_msg6.pose.orientation.x = 0
                pos_msg6.pose.orientation.y = 0
                pos_msg6.pose.orientation.z = 0
                pos_msg6.pose.orientation.w = 0
                self.pub6.publish(pos_msg6)

                if i < self.data.shape[0] - 1:
                    i += 1

                self.rate.sleep()
                time.sleep(0.1)


if __name__ == '__main__':
    try:
        x = Drone()
        x.detection()
        x.random_walk()

    except rospy.ROSInterruptException:
        pass