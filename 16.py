import cv2
from matplotlib import pyplot as plt

image = cv2.imread('flower.jpeg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image, 100, 200)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')
plt.show()
