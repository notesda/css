import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_text_watermark(file, out, mark, size, color, opacity, angle, space):
    # Load the image
    image = cv2.imread(file)
    (h, w) = image.shape[:2]
    
    # Prepare the watermark overlay
    overlay = np.zeros((h, w, 3), dtype="uint8")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = size / 100
    color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    thickness = int(size / 20)
    
    (text_w, text_h), baseline = cv2.getTextSize(mark, font, scale, thickness)
    text_h += baseline
    
    for y in range(0, h, text_h + space):
        for x in range(0, w, text_w + space):
            cv2.putText(overlay, mark, (x, y), font, scale, color_bgr, thickness, cv2.LINE_AA)
    
    if angle != 0:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        overlay = cv2.warpAffine(overlay, M, (w, h))
    
    watermarked_image = cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
    cv2.imwrite(out, watermarked_image)
    
    # Convert images for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    watermarked_image_rgb = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB)
    
    # Display the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Watermarked Image')
    plt.imshow(watermarked_image_rgb)
    plt.axis('off')
    
    plt.show()

# Example usage
add_text_watermark(
    'sunflower.jpg', 
    'watermarked_image.png', 
    'UVCE', 
    80, 
    '#ffffff', 
    0.2, 
    30, 
    40
)
