# -*- coding: utf-8 -*-
import cv2
import numpy as np
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

def average(write_img):
    y = cv2.getTrackbarPos("Gradient", "img")
    grad = img[y, 0:w]
    sum = 0
    set = 5
    next_grad = np.zeros(len(grad))
    for i in range(len(grad)):
        sum += grad[i][0]
        if i < set:
            next_grad[i] = sum / (i + 1)
        else:
            sum-=grad[i-set][0]
            next_grad[i] = sum / set

    grad_img = np.zeros((256, w, 3), np.uint8)
    for i in range(len(next_grad)):
        grad_img[255 - int(next_grad[i]), i] = (255, 255, 255)
    print("write!")

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
    sum = 0
    set = 15
    round_grad = np.zeros(len(grad))
    for i in range(len(grad)):
        sum += grad[i][0]
        if i < set:
            round_grad[i] = sum / (i + 1)
        else:
            sum -= grad[i - set][0]
            round_grad[i] = sum / set

    grad_img = np.zeros((256, w, 3), np.uint8)
    for i in range(len(round_grad)):
        grad_img[255 - int(round_grad[i]), i] = (255, 255, 255)
        if 0 < i and i < len(round_grad)-1 and round_grad[i - 1] <= round_grad[i] and round_grad[i] >= round_grad[i + 1]:
            rut_img[y, i] = (255, 0, 0)


cv2.destroyAllWindows()
