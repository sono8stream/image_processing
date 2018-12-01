#!/usr/bin/env python
# -*- coding: utf-8 -*-
#2018 Hasegawa feat Morii san
#for /thermal_mono
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import UInt32
from cv_bridge import CvBridge, CvBridgeError


class ImagePublisher(object):
    def __init__(self):
        self._image_pub = rospy.Publisher('/edge_image', Image, queue_size=1)
        self._mono_sub = rospy.Subscriber('/bird_eye_image', Image, self.callbackmono)
        self._bridge = CvBridge()

    def callbackmono(self, data):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError, e:
            print e

        pub_img = hough_lines(cv_image)

        try:
            self._image_pub.publish(
                self._bridge.cv2_to_imgmsg(pub_img, 'bgr8'))

        except CvBridgeError, e:
            print e


def hough_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([0, 0, 255]))

    gray_sobelx = cv2.Sobel(img_mask, cv2.CV_32F, 1, 0, ksize=3)
    gray_sobely = cv2.Sobel(img_mask, cv2.CV_32F, 0, 1, ksize=3)
    gray_abs_x = cv2.convertScaleAbs(gray_sobelx)
    gray_abs_y = cv2.convertScaleAbs(gray_sobely)
    gray_sobel_edge = cv2.addWeighted(gray_abs_x, 0.5, gray_abs_y, 0.5, 0)

    lines = cv2.HoughLines(gray_sobel_edge, 1, np.pi / 180, 200)
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    if lines is None:
        return img_mask

    for line in lines:
        rho, theta = line[0]
        if theta < np.pi - 0.2 and theta > 0.2:
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img_mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img_mask

if __name__ == '__main__':
    rospy.init_node('RutImagePublisher')
    color = ImagePublisher()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

