import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('flower.jpeg', 0)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)

prewitt_kernel_x = np.array([[-1, 0, 1],
[-1, 0, 1],
[-1, 0, 1]])
prewitt_kernel_y = np.array([[-1, -1, -1],
[0, 0, 0],
[1, 1, 1]])
prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y)
prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)

laplacian = cv2.Laplacian(image, cv2.CV_64F)

plt.figure(figsize=(12, 8))
plt.subplot(221), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(222), plt.imshow(sobel, cmap='gray'), plt.title('Sobel Filter')
plt.subplot(223), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt Filter')
plt.subplot(224), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian Filter')
plt.show()
