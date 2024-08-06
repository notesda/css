import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('flower.jpeg', 0)

laplacian = cv2.Laplacian(image, cv2.CV_64F)

sharpened_image = np.clip(image - 0.5*laplacian, 0, 255).astype(np.uint8)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(sharpened_image, cmap='gray'), plt.title('Sharpened Image')
plt.show()
