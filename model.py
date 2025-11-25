import torch
import torch.nn as nn
import torchvision.models as models


class MultiModalDetector(nn.Module):
    """
    Multi-Modal Deepfake Detector
    Combines RGB, Frequency, and Noise features
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super(MultiModalDetector, self).__init__()
        
        # ========================================
        # RGB Stream (ResNet50)
        # ========================================
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the final FC layer
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])
        rgb_features = 2048
        
        # ========================================
        # Frequency Stream (Simple CNN)
        # ========================================
        self.freq_stream = nn.Sequential(
            # Input: (1, 256, 256)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling
        )
        freq_features = 256
        
        # ========================================
        # Noise Stream (Simple CNN)
        # ========================================
        self.noise_stream = nn.Sequential(
            # Input: (3, 256, 256)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        noise_features = 256
        
        # ========================================
        # Fusion Layer
        # ========================================
        total_features = rgb_features + freq_features + noise_features
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, rgb, frequency, noise):
        """
        Args:
            rgb: (B, 3, H, W)
            frequency: (B, 1, H, W)
            noise: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        # Extract features from each stream
        rgb_feat = self.rgb_backbone(rgb)  # (B, 2048, 1, 1)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)  # (B, 2048)
        
        freq_feat = self.freq_stream(frequency)  # (B, 256, 1, 1)
        freq_feat = freq_feat.view(freq_feat.size(0), -1)  # (B, 256)
        
        noise_feat = self.noise_stream(noise)  # (B, 256, 1, 1)
        noise_feat = noise_feat.view(noise_feat.size(0), -1)  # (B, 256)
        
        # Concatenate all features
        combined = torch.cat([rgb_feat, freq_feat, noise_feat], dim=1)  # (B, 2560)
        
        # Final classification
        output = self.fusion(combined)  # (B, num_classes)
        
        return output


# Test the model
if __name__ == "__main__":
    # Create dummy inputs
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 256, 256)
    frequency = torch.randn(batch_size, 1, 256, 256)
    noise = torch.randn(batch_size, 3, 256, 256)
    
    # Create model
    model = MultiModalDetector(num_classes=2, pretrained=False)
    
    # Forward pass
    output = model(rgb, frequency, noise)
    
    print("✅ Model Test Successful!")
    print(f"Input shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Frequency: {frequency.shape}")
    print(f"  Noise: {noise.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")