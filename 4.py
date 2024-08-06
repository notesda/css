import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_image(image, title, subplot_position):
    plt.subplot(3, 3, subplot_position)
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

def bit_plane_slice(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    bit_planes = []
    for i in range(8):
        bit_plane = (gray_image >> i) & 1
        bit_planes.append(bit_plane)
    display_image(image, 'Original Image', 1)
    for i, bit_plane in enumerate(bit_planes):
        display_image(bit_plane * 255, f'Bit Plane {i}', i+2)  

image_path = '1.jpg'
image = cv2.imread(image_path)


bit_plane_slice(image)

plt.tight_layout()
plt.show()
