import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('flower.jpeg', 0)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(image, kernel, iterations=1)

dilation = cv2.dilate(image, kernel, iterations=1)

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 8))
plt.subplot(321), plt.imshow(image, cmap='gray'), plt.title('Original Image'),plt.axis('off')
plt.subplot(322), plt.imshow(erosion,cmap='gray'), plt.title('Erosion'),plt.axis('off')
plt.subplot(323), plt.imshow(dilation, cmap='gray'), plt.title('Dilation'),plt.axis('off')
plt.subplot(324), plt.imshow(opening, cmap='gray'), plt.title('Opening'),plt.axis('off')
plt.subplot(325), plt.imshow(closing, cmap='gray'), plt.title('Closing'),plt.axis('off')
plt.show()
