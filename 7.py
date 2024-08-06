import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

kernel_size = 3

kernel_lp = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size* kernel_size)
smoothed_image_lp = cv2.filter2D(image, -1, kernel_lp)

kernel_hp = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
smoothed_image_hp = cv2.filter2D(image, -1, kernel_hp)

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(smoothed_image_lp, cmap='gray'), plt.title('Low-pass Filtered Image')
plt.subplot(133), plt.imshow(smoothed_image_hp, cmap='gray'), plt.title('High-pass Filtered Image')
plt.show()
