import cv2
import numpy as np
import sys

args = sys.argv
img = cv2.imread(args[1])
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

def nothing(x):
  pass

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.createTrackbar("lower", "img", 0, 255, nothing)
cv2.createTrackbar("upper", "img", 0, 255, nothing)

img_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([0, 0, 255]))

while (True):
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  lower = cv2.getTrackbarPos("lower", "img")
  upper = cv2.getTrackbarPos("upper", "img")
  lower_color = np.array([0,0, lower])
  upper_color = np.array([0,0, upper])

  img_mask = cv2.inRange(hsv, lower_color, upper_color)

  cv2.imshow("img", img_mask)

cv2.imwrite('mask.png',img_mask)
