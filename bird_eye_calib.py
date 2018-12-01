import cv2
import numpy as np
import matplotlib.pyplot as plt

poses = np.array([])
click_cnt = 0

IMAGE_H = 251
IMAGE_W = 333
IMAGE_UP = 180

def mouse_event(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    global poses
    global click_cnt
    poses = np.append(poses, (x, y))
    click_cnt = click_cnt + 1
    print(click_cnt)

# Read the test img
img = cv2.imread('./thermal_mono_screenshot_31.10.2018.png')

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("img", mouse_event)

while (True):
  cv2.imshow("img",img)
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break

cv2.destroyAllWindows()
