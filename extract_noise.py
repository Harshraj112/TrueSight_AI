import cv2
import numpy as np

def extract_noise_map(image_path):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))

    # Apply Gaussian blur (smooths the image)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Subtract to get noise
    noise = img - blurred

    return noise

# Test it
noise_map = extract_noise_map('dataset/real/real_00001.jpg')
print(noise_map.shape)  # Should be (256, 256, 3)