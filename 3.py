import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_image(image, title, subplot_position):
    plt.subplot(2, 3, subplot_position)
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

Original_image = cv2.imread('flower.jpeg')
image = cv2.imread('flower.jpeg', cv2.IMREAD_GRAYSCALE)

negative_image = np.uint8(255 - image)
    
c = 255 / np.log(1 + np.max(image))
log_transformed_image = np.uint8(c * (np.log(image + 1e-5)))
    
gamma = 1.1
power_law_transformed_image = np.uint8(np.power(image, gamma))
    
min_intensity = np.min(image)
max_intensity = np.max(image)
contrast_stretched_image = np.uint8(255 * ((image - min_intensity) / (max_intensity - min_intensity)))
    
display_image(Original_image, 'Original', 1)
display_image(negative_image, 'Negative', 2)
display_image(log_transformed_image, 'Log Transform', 3)
display_image(power_law_transformed_image, 'Power Law Transform', 4)
display_image(contrast_stretched_image, 'Contrast Stretched', 5)
    
plt.tight_layout()
plt.show()
