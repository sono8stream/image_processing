import cv2
import matplotlib
print(matplotlib.__version__)
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

m = 2  # dimension
mean = np.zeros(m)
sigma = np.eye(m)

N = 1000
x1 = np.linspace(-5, 5, N)
x2 = np.linspace(-5, 5, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]
Y_plot = multivariate_normal.pdf(x=X, mean=mean, cov=sigma)
Y_plot = Y_plot.reshape(X1.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contour(X1, X2, Y_plot)
ax.set_title("Contour Plot")
fig.show()

input()
