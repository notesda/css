import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('1.jpg')

plt.subplot(3, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(3, 3, 2)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

b, g, r = cv2.split(image)
plt.subplot(3, 3, 3)
plt.title('Red Channel')
plt.imshow(r, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 4)
plt.title('Green Channel')
plt.imshow(g, cmap='gray')
plt.axis('off')
plt.subplot(3, 3, 5)
plt.title('Blue Channel')
plt.imshow(b, cmap='gray')
plt.axis('off')

kernel_1d = np.array([1, 0, -1])
conv_1d_image = cv2.filter2D(gray_image, -1, kernel_1d)
plt.subplot(3, 3, 6)
plt.title('1D Convolution')
plt.imshow(conv_1d_image, cmap='gray')
plt.axis('off')

kernel_2d = np.array([[1, 1, 1],
[1, -8, 1],
[1, 1, 1]])
conv_2d_image = cv2.filter2D(gray_image, -1, kernel_2d)
plt.subplot(3, 3, 7)
plt.title('2D Convolution')
plt.imshow(conv_2d_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
