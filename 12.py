import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('flower.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixels = image.reshape((-1, 3))

pixels = np.float32(pixels)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3 # Number of clusters
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10,
cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)

segmented_image = centers[labels.flatten()]

segmented_image = segmented_image.reshape(image.shape)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image), plt.title('Original Image')
plt.subplot(122),plt.imshow(segmented_image), plt.title('Segmented Image')
plt.show()
