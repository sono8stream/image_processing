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
        self._image_pub = rospy.Publisher('/bird_eye_image', Image, queue_size=1)
        self._mono_sub = rospy.Subscriber('/thermal_mono', Image, self.callbackmono)
        self._bridge = CvBridge()

    def callbackmono(self, data):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError, e:
            print e

        pub_img = bird_eye(cv_image)

        try:
            self._image_pub.publish(
                self._bridge.cv2_to_imgmsg(pub_img, 'bgr8'))

        except CvBridgeError, e:
            print e


def bird_eye(img):
    IMAGE_H = 287
    IMAGE_W = 381
    IMAGE_UP = 200
    IMAGE_SIDE = 140

    src = np.float32([[0, IMAGE_H - IMAGE_UP],[IMAGE_W, IMAGE_H - IMAGE_UP], [0, 0], [IMAGE_W, 0]])  # もと画像から抽出する範囲
    dst = np.float32([[IMAGE_SIDE, IMAGE_H],
        [IMAGE_W - IMAGE_SIDE, IMAGE_H], [0, 0], [IMAGE_W, 0]])  # 目標描画形状
    M = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix

# Read the test img
    img = img[IMAGE_UP:IMAGE_H, 0:IMAGE_W]  # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(
        img, M, (IMAGE_W, IMAGE_H))  # Image warping
    hsv = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)
    img_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([0, 0, 255]))
    img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    return img_mask

if __name__ == '__main__':
    rospy.init_node('BirdEyeImagePublisher')
    color = ImagePublisher()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

