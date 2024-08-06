import cv2
from matplotlib import pyplot as plt

noisy_img = cv2.imread('1.jpg', 0) 
denoised_img = cv2.medianBlur(noisy_img, 5)

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(noisy_img, cmap='gray'), plt.title('Noise image')
plt.subplot(122), plt.imshow(denoised_img, cmap='gray'), plt.title('Denoised Image')
plt.show()
