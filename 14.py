import cv2
import matplotlib.pyplot as plt

input_image_path = '1.jpg'
input_image = cv2.imread(input_image_path)

grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

kernel_size = 3
restored_image = cv2.medianBlur(grayscale_image, kernel_size)

input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(input_image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(restored_image, cmap='gray')
plt.title('Restored Image')
plt.axis('off')

plt.show()
