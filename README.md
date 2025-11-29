# 🧠 TrueSight_AI – Multi-Modal Deepfake Detection System  
### RGB + Noise + Frequency Feature Fusion for Real vs Fake Image Classification

##[If you want dataset for this please download](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

# 🔍 TrueSight_AI — Multi-Modal Deepfake Detection

TrueSight_AI is a **three-branch deepfake detection system** that analyzes:
- **RGB domain**
- **Noise (PRNU) domain**
- **Frequency (FFT) domain**

This multi-modal design catches deepfakes far more reliably than RGB alone.

---

# 📊 System Flowchart

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/117c4ff3-3bb3-4c2e-85b7-f930a4d3c21c" />
`flowchart.png`


<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/5dfabb41-c04e-440d-9003-9d4ba0443bf0" />


## 📘 1. Overview

Deepfake detection is hard using only RGB pixels.

Different domains reveal **different hidden clues**:

| Domain      | What It Detects | Why It Works |
|-------------|------------------|---------------|
| **RGB**     | Visual noise, textures, shadows | GANs hallucinate micro-details |
| **Noise**   | Missing PRNU camera fingerprints | AI images lack real sensor noise |
| **Frequency** | Checkerboard patterns, FFT spikes | GANs leave periodic artifacts |

**TrueSight_AI** extracts all three and merges them into a unified classifier.

---

## 🧬 2. Theory Behind Each Branch

### 🔴 **RGB Branch**
A CNN extracts:
- Texture details  
- Color patterns  
- Edge behavior  
- Micro-structure inconsistencies  

GANs often miss tiny high-frequency structures.

---

### 🟣 **Noise Branch (PRNU Residuals)**

Noise is extracted using:
Noise = Image - GaussianBlur(Image)

This branch detects:
- Missing camera sensor noise  
- Over-smoothness  
- Denoising artifacts  
- Fake PRNU patterns  

Real images always contain PRNU → AI images don’t.

---

### 🔵 **Frequency Branch (FFT Features)**

FFT reveals:
- High-frequency spikes  
- Checkerboard artifacts  
- Moiré patterns  
- Periodic GAN noise  

These are **invisible in RGB**, but obvious in frequency space.

---

## 🏗 3. Project Structure



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



---

## ⚙️ 4. Installation

```bash
pip install torch torchvision opencv-python pillow numpy matplotlib tqdm seaborn scikit-learn
```

## 🧪 5. Verify Setup

Run:

```bash
python verify_setup.py
```

This script verifies that your environment is ready by checking:

📁 Dataset paths
🔧 Model imports
⚡ GPU / MPS / CPU availability
🖼️ Image loader integrity

## 🏋️ 6. Train the Model

python train.py
best_model.pth
training_curves.png
confusion_matrix.png
metrics.json

## 🧾 7. Evaluate Full Test Dataset

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

## 📊 8. Generated Outputs

Training curve
Confusion matrix
Real vs Fake confidence
Per-image prediction logs

## 🚀 9. Future Enhancements

Add Vision Transformer (ViT) branch
Add Wavelet Transform (DWT)
Integrate CLIP features
Deploy using FastAPI or Streamlit
