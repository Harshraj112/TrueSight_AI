import cv2
import numpy as np
from numpy.fft import fft2, fftshift
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

class FeatureExtractor:
    """
    Extract three types of features from images:
    1. RGB Image (original)
    2. Frequency Map (FFT)
    3. Noise Residual Map
    """
    
    def __init__(self, image_size=256):
        self.image_size = image_size
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_rgb(self, image_path):
        """
        Load and preprocess RGB image
        Returns: tensor of shape (3, H, W)
        """
        img = Image.open(image_path).convert('RGB')
        return self.transform(img)
    
    def extract_frequency_map(self, image_path):
        """
        Extract frequency domain features using FFT
        Returns: tensor of shape (1, H, W)
        """
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Apply 2D FFT
        fft = fft2(img)
        fft_shift = fftshift(fft)  # Shift zero frequency to center
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_shift)
        
        # Apply log transform for better visualization
        magnitude = np.log(magnitude + 1)
        
        # Normalize to [0, 1]
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        # Convert to tensor
        magnitude = torch.from_numpy(magnitude).float().unsqueeze(0)
        
        return magnitude
    
    def extract_noise_map(self, image_path):
        """
        Extract noise residual using High-Pass Filter
        Returns: tensor of shape (3, H, W)
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply Gaussian blur (Low-Pass Filter)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Subtract to get high-frequency noise
        noise = cv2.subtract(img, blurred)
        
        # Normalize
        noise = noise.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        noise = torch.from_numpy(noise).permute(2, 0, 1).float()
        
        return noise
    
    def extract_all_features(self, image_path):
        """
        Extract all three features at once
        Returns: dict with 'rgb', 'frequency', 'noise'
        """
        return {
            'rgb': self.extract_rgb(image_path),
            'frequency': self.extract_frequency_map(image_path),
            'noise': self.extract_noise_map(image_path)
        }


# Test the extractor
if __name__ == "__main__":
    extractor = FeatureExtractor(image_size=256)
    
    # Test with a sample image
    test_image = "dataset/real/real_00001.jpg"  # Replace with your path
    
    try:
        features = extractor.extract_all_features(test_image)
        print("✅ Feature Extraction Successful!")
        print(f"RGB shape: {features['rgb'].shape}")
        print(f"Frequency shape: {features['frequency'].shape}")
        print(f"Noise shape: {features['noise'].shape}")
    except Exception as e:
        print(f"❌ Error: {e}")