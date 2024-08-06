import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft_convolve(img, kernel):
    fshift = np.fft.fftshift(np.fft.fft2(img))
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift * kernel))
    return np.abs(img_back)

img = cv2.imread('flower.jpeg', 0)
rows, cols = img.shape
center = (rows // 2, cols // 2)
radius = 30

mask_lp = np.zeros((rows, cols), np.uint8)
cv2.circle(mask_lp, center, radius, 1, -1)
mask_hp = np.ones((rows, cols), np.uint8)
cv2.circle(mask_hp, center, radius, 0, -1)

smoothed_img_lp = fft_convolve(img, mask_lp)
smoothed_img_hp = fft_convolve(img, mask_hp)

plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(smoothed_img_lp, cmap='gray'), plt.title('Low-pass Filter')
plt.subplot(133), plt.imshow(smoothed_img_hp, cmap='gray'), plt.title('High-pass Filter')
plt.show()
