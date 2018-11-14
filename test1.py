import cv2
import matplotlib as mpl
mpl.use('tkagg')
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
