import cv2
import matplotlib.pyplot as plt

image=cv2.imread('1.jpg')
gray_image=cv2.imread('1.jpg', 0)

equalized_image = cv2.equalizeHist (gray_image)

plt.figure(figsize=(10,5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')
plt.subplot(1, 3, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.tight_layout()
plt.show()
