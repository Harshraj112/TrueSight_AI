# 🧠 TrueSight_AI – Multi-Modal Deepfake Detection System  
### RGB + Noise + Frequency Feature Fusion for Real vs Fake Image Classification

TrueSight_AI is an advanced deepfake detection architecture that analyzes images in **three domains**:

- **RGB Domain** → visual artifacts  
- **Noise Domain (PRNU Residuals)** → missing camera sensor noise  
- **Frequency Domain (FFT Patterns)** → GAN fingerprint patterns  

By combining these, TrueSight_AI achieves far higher accuracy than single-branch CNN models.

---

# 📊 System Flowchart

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/fa7ccd1e-6b2a-47d2-bd08-cf0e428688fe" />
 
`flowchart.png`
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/5dfabb41-c04e-440d-9003-9d4ba0443bf0" />


📘 1. Overview
Deepfake detection is hard using RGB alone. AI-generated images leave hidden clues in:

Domain	What It Detects	Why It Works
RGB	Visual noise, textures, shadows	GANs hallucinate micro-details
Noise	Missing PRNU	No real camera sensor → fake
Frequency	Checkerboard patterns, FFT spikes	Strong GAN artifacts

TrueSight_AI extracts all three representations and merges them into a multi-modal classifier.

🧬 2. Theory Behind Each Branch
🔴 RGB Branch
CNN extracts:

Texture detail

Color patterns

Edge behavior

Micro-structure inconsistencies

🟣 Noise Branch (PRNU Residuals)

Noise = Image - GaussianBlur(Image)
Real cameras contain PRNU → AI-generated images do not.
Noise branch detects:
Missing sensor patterns
Over-smoothing
Bad denoising artifacts

🔵 Frequency Branch (FFT Features)
FFT highlights hidden GAN artifacts:

High-frequency spikes
Moiré patterns
Checkerboard artifacts
Periodic GAN residuals
These patterns are invisible in RGB but obvious in frequency domain.

🏗 3. Project Structure
go
Copy code
TrueSight_AI/
│── model.py
│── train.py
│── test_model.py
│── predict_single.py
│── verify_setup.py
│── dataset.py
│── feature_extractor.py
│── saved/
│    ├── best_model.pth
│    ├── training_curves.png
│    ├── confusion_matrix.png
│── dataset/
     ├── train/
     │    ├── real/
     │    └── fake/
     └── test/
          ├── real/
          └── fake/


⚙️ 4. Installation
pip install torch torchvision opencv-python pillow numpy matplotlib tqdm seaborn scikit-learn

🧪 5. Verify Setup
python verify_setup.py

Checks:
Dataset exists
Model imports
GPU availability
Image loaders

🏋️ 6. Train the Model
python train.py
best_model.pth
training_curves.png
confusion_matrix.png
metrics.json

🧾 7. Evaluate Full Test Dataset
bash
Copy code
python test_model.py
Outputs:

Accuracy
Precision
Recall
F1-Score
Confusion matrix
Wrong predictions

📊 8. Generated Outputs

Training curve
Confusion matrix
Real vs Fake confidence
Per-image prediction logs

🚀 9. Future Enhancements
Add Vision Transformer (ViT) branch
Add Wavelet Transform (DWT)
Integrate CLIP features
Deploy using FastAPI or Streamlit
