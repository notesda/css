import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_image(image, title, subplot_position):
    plt.subplot(4, 3, subplot_position)
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

image1 = cv2.imread('1.jpg')
image2 = cv2.imread('flower.jpeg')

height, width, _ = image1.shape
image2 = cv2.resize(image2, (width, height))

display_image(image1, 'Original Image1', 1)
display_image(image2, 'Original Image2', 2)

added_image = cv2.add(image1, image2) 
subtracted_image = cv2.subtract(image1, image2)  
multiplied_image = cv2.multiply(image1, image2)  

image2_float32 = image2.astype(float)
image2_float32[image2_float32 == 0] = 1 
divided_image = cv2.divide(image1.astype(float), image2_float32)

divided_image_uint8 = np.clip(divided_image * 255, 0, 255).astype('uint8')

display_image(added_image, 'Added Image', 3)
display_image(subtracted_image, 'Subtracted Image', 4)
display_image(multiplied_image, 'Multiplied Image', 5)
display_image(divided_image_uint8, 'Divided Image', 6)

image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(image1_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(image2_gray, 127, 255, cv2.THRESH_BINARY)

bitwise_and = cv2.bitwise_and(thresh1, thresh2)
bitwise_or = cv2.bitwise_or(thresh1, thresh2)
bitwise_xor = cv2.bitwise_xor(thresh1, thresh2)
bitwise_not1 = cv2.bitwise_not(thresh1)
bitwise_not2 = cv2.bitwise_not(thresh2)

display_image(bitwise_and, 'Bitwise AND', 7)
display_image(bitwise_or, 'Bitwise OR', 8)
display_image(bitwise_xor, 'Bitwise XOR', 9)
display_image(bitwise_not1, 'Bitwise NOT1', 10)
display_image(bitwise_not2, 'Bitwise NOT2', 11)

plt.tight_layout()
plt.show()
