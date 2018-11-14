import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_H = 251
IMAGE_W = 333
IMAGE_UP = 180

src = np.float32([[0, IMAGE_H-IMAGE_UP], [IMAGE_W, IMAGE_H-IMAGE_UP], [0, 0], [IMAGE_W, 0]])# もと画像から抽出する範囲
dst = np.float32([[150, IMAGE_H], [182, IMAGE_H], [0, 0], [IMAGE_W, 0]])# 目標描画形状
M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

# Read the test img
img = cv2.imread('./thermal_mono_screenshot_31.10.2018.png')
img = img[IMAGE_UP:IMAGE_H, 0:IMAGE_W] # Apply np slicing for ROI crop
warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H-IMAGE_UP)) # Image warping
#plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
#plt.show()

def nothing(x):
  pass


cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.createTrackbar("bottom", "img", 0, (int)(IMAGE_W/2), nothing)

while (True):
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  v = cv2.getTrackbarPos("bottom", "img")
  dst = np.float32([[v, IMAGE_H],
                    [IMAGE_W-v, IMAGE_H],
                    [0, 0], [IMAGE_W, 0]])  # 目標描画形状更新
  M=cv2.getPerspectiveTransform(src,dst)
  warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))  # Image warping
  cv2.imshow("img", warped_img)

cv2.imwrite('bird_eye.png', warped_img)
