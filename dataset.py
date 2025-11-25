import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from feature_extractor import FeatureExtractor

class ImageDetectorDataset(Dataset):
    """
    Custom Dataset for Real vs Fake Image Detection
    Returns: RGB, Frequency Map, Noise Map, Label
    """
    
    def __init__(self, data_dir, split='train', image_size=256):
        """
        Args:
            data_dir: Root directory containing 'real' and 'fake' folders
            split: 'train' or 'test'
            image_size: Size to resize images to
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.extractor = FeatureExtractor(image_size=image_size)
        
        # Collect all image paths and labels
        self.samples = []
        self.labels = []
        
        # Real images (label = 0)
        real_dir = self.data_dir / split / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*.[jp][pn]g'):  # .jpg, .png, .jpeg
                self.samples.append(img_path)
                self.labels.append(0)
        
        # Fake images (label = 1)
        fake_dir = self.data_dir / split / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.[jp][pn]g'):
                self.samples.append(img_path)
                self.labels.append(1)
        
        print(f"✅ Loaded {len(self.samples)} images from {split} set")
        print(f"   Real: {self.labels.count(0)}, Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            rgb: (3, H, W) tensor
            frequency: (1, H, W) tensor
            noise: (3, H, W) tensor
            label: 0 or 1
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            # Extract all features
            features = self.extractor.extract_all_features(img_path)
            
            return (
                features['rgb'],
                features['frequency'],
                features['noise'],
                torch.tensor(label, dtype=torch.long)
            )
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__((idx + 1) % len(self))


def get_dataloaders(data_dir, batch_size=16, num_workers=4, image_size=256):
    """
    Create train and validation dataloaders
    """
    # Create datasets
    train_dataset = ImageDetectorDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size
    )
    
    test_dataset = ImageDetectorDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# Test the dataset
if __name__ == "__main__":
    # Test loading
    data_dir = "dataset"  # Your dataset path
    
    try:
        train_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0  # Use 0 for debugging
        )
        
        # Test one batch
        for rgb, freq, noise, labels in train_loader:
            print(f"\n✅ Batch loaded successfully!")
            print(f"RGB shape: {rgb.shape}")
            print(f"Frequency shape: {freq.shape}")
            print(f"Noise shape: {noise.shape}")
            print(f"Labels: {labels}")
            break
            
    except Exception as e:
        print(f"❌ Error: {e}")