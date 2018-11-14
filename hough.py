import cv2
import numpy as np
import sys

args = sys.argv
img = cv2.imread(args[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
cv2.imshow('gray_sobelx', gray_sobelx)

gray_sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
cv2.imshow('gray_sobely', gray_sobely)

gray_abs_x = cv2.convertScaleAbs(gray_sobelx)
gray_abs_y = cv2.convertScaleAbs(gray_sobely)

gray_sobel_edge = cv2.addWeighted(gray_abs_x, 0.5, gray_abs_y, 0.5, 0)
cv2.imshow('gray_sobel_edge', gray_sobel_edge)

cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLines(gray_sobel_edge, 1, np.pi / 180, 300)

if lines == None:
  print('Lines None!')
  sys.exit()

print(len(lines))
for line in lines:
    rho, theta = line[0]
    if theta < np.pi - 0.05 and theta > 0.05:
      continue
    print(theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('houghlines3.png',img)
