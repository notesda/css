import cv2
import numpy as np

def block_truncation_coding(input_image_path, output_image_path, block_size=8):
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    compressed_image = np.zeros_like(image)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            block_mean = np.mean(block)
            compressed_image[i:i+block_size, j:j+block_size] = np.where(block <= block_mean, 0, 255)
    
    cv2.imwrite(output_image_path, compressed_image)

# Example usage
block_truncation_coding('flower.jpeg', 'compressed_image.jpg', block_size=8)
