# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy import signal
import sys

# 画像の読み込み
args = sys.argv
img = cv2.imread(args[1])
rut_img=cv2.imread(args[1])
h, w, c = img.shape
print(w)
print(h)
print(c)

# edge=cv2.Canny(img,0,0)
# ウィンドウのサイズを変更可能にする
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.namedWindow("grad_img", cv2.WINDOW_NORMAL)

grad_img = np.zeros((256, w, 3), np.uint8)

def nothing(x):
    pass

# トラックバーの生成
cv2.createTrackbar("Min", "img", 0, 255, nothing)
cv2.createTrackbar("Max", "img", 0, 255, nothing)
cv2.createTrackbar("Gradient","img",0,h-1,nothing)


# 「Q」が押されるまで画像を表示する
while (True):
    cv2.imshow("img", rut_img)
    cv2.imshow("grad_img",grad_img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    min = cv2.getTrackbarPos("Min", "img")
    max = cv2.getTrackbarPos("Max", "img")
    edge = cv2.Canny(img, min, max)

    y = cv2.getTrackbarPos("Gradient", "img")
    grad = img[y, 0:w]
    mono_grad = np.zeros(len(grad))
    for i in range(len(grad)):
        mono_grad[i] = grad[i][0]

#極大検出
    grad_img = np.zeros((256, w, 3), np.uint8)
    maxids = signal.argrelmax(mono_grad, order=10)
    if len(maxids[0]) == 0: continue

    for i in range(len(maxids[0])):
        grad_img[255 - grad[maxids[0][i]][0], maxids[0][i]] = (255, 255, 255)
        rut_img[y, maxids[0][i]] = (255, 0, 0)


cv2.destroyAllWindows()
