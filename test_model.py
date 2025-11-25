"""
Test the trained model on test dataset
Generates detailed evaluation report with visualizations
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from pathlib import Path
from tqdm import tqdm
import json

from model import MultiModalDetector
from dataset import ImageDetectorDataset


class ModelTester:
    def __init__(self, model_path, data_dir, device, save_dir='test_results'):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Load model
        print("📥 Loading model...")
        self.model = MultiModalDetector(num_classes=2, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"✅ Model loaded from: {model_path}")
        print(f"   Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
        
        # Load test dataset
        print("\n📊 Loading test dataset...")
        self.test_dataset = ImageDetectorDataset(
            data_dir=data_dir,
            split='test',
            image_size=256
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def predict(self):
        """Run predictions on entire test set"""
        print("\n🔮 Running predictions...")
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for rgb, freq, noise, labels in tqdm(self.test_loader, desc="Testing"):
                # Move to device
                rgb = rgb.to(self.device)
                freq = freq.to(self.device)
                noise = noise.to(self.device)
                
                # Get predictions
                outputs = self.model(rgb, freq, noise)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of "fake"
                all_labels.extend(labels.numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def calculate_metrics(self, labels, preds, probs):
        """Calculate all evaluation metrics"""
        print("\n📈 Calculating metrics...")
        
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='binary'),
            'recall': recall_score(labels, preds, average='binary'),
            'f1_score': f1_score(labels, preds, average='binary'),
        }
        
        # Per-class metrics
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def print_results(self, metrics):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("                    TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        
        print(f"\n🎯 Detailed Metrics:")
        print(f"   True Positives (Correctly detected fakes):  {metrics['true_positives']}")
        print(f"   True Negatives (Correctly identified real): {metrics['true_negatives']}")
        print(f"   False Positives (Real marked as fake):      {metrics['false_positives']}")
        print(f"   False Negatives (Fake marked as real):      {metrics['false_negatives']}")
        
        print(f"\n📉 Error Rates:")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate']*100:.2f}%)")
        print(f"   False Negative Rate: {metrics['false_negative_rate']:.4f} ({metrics['false_negative_rate']*100:.2f}%)")
        print(f"   Specificity:         {metrics['specificity']:.4f}")
        
        print("\n" + "="*60)
    
    def plot_confusion_matrix(self, labels, preds):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Real', 'Fake'], 
                    yticklabels=['Real', 'Fake'],
                    cbar_kws={'label': 'Count'})
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        save_path = self.save_dir / 'confusion_matrix_test.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, labels, probs):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / 'roc_curve_test.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved to: {save_path}")
        plt.close()
        
        return roc_auc
    
    def plot_prediction_distribution(self, labels, probs):
        """Plot distribution of prediction probabilities"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Separate probabilities by true label
        real_probs = probs[labels == 0]
        fake_probs = probs[labels == 1]
        
        # Plot 1: Histograms
        ax1.hist(real_probs, bins=50, alpha=0.6, label='Real Images', color='green')
        ax1.hist(fake_probs, bins=50, alpha=0.6, label='Fake Images', color='red')
        ax1.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        ax1.set_xlabel('Predicted Probability (Fake)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Distribution of Prediction Probabilities', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Box plots
        data_to_plot = [real_probs, fake_probs]
        bp = ax2.boxplot(data_to_plot, labels=['Real', 'Fake'], 
                         patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][1].set_facecolor('red')
        ax2.axhline(y=0.5, color='black', linestyle='--', label='Decision Threshold')
        ax2.set_ylabel('Predicted Probability (Fake)', fontsize=11)
        ax2.set_title('Probability Distribution by True Label', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / 'prediction_distribution_test.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Prediction distribution saved to: {save_path}")
        plt.close()
    
    def save_results(self, metrics, roc_auc):
        """Save all metrics to JSON"""
        results = {
            **metrics,
            'roc_auc': float(roc_auc),
            'total_samples': len(self.test_dataset),
        }
        
        save_path = self.save_dir / 'test_metrics.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✅ Metrics saved to: {save_path}")
    
    def generate_classification_report(self, labels, preds):
        """Generate and save detailed classification report"""
        report = classification_report(
            labels, preds,
            target_names=['Real', 'Fake'],
            digits=4
        )
        
        print("\n📋 Classification Report:")
        print(report)
        
        # Save to file
        save_path = self.save_dir / 'classification_report.txt'
        with open(save_path, 'w') as f:
            f.write("Classification Report - Test Set\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        print(f"✅ Classification report saved to: {save_path}")
    
    def analyze_errors(self, labels, preds, probs):
        """Analyze misclassified samples"""
        print("\n🔍 Analyzing errors...")
        
        # Find misclassified samples
        errors = labels != preds
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            print("🎉 Perfect! No errors found!")
            return
        
        # Separate types of errors
        false_positives = np.where((labels == 0) & (preds == 1))[0]
        false_negatives = np.where((labels == 1) & (preds == 0))[0]
        
        print(f"\n❌ Total Errors: {len(error_indices)} / {len(labels)} ({len(error_indices)/len(labels)*100:.2f}%)")
        print(f"   False Positives: {len(false_positives)} (Real images marked as Fake)")
        print(f"   False Negatives: {len(false_negatives)} (Fake images marked as Real)")
        
        # Most confident errors
        error_probs = probs[errors]
        error_labels = labels[errors]
        
        # For false positives: high probability of fake (close to 1)
        # For false negatives: low probability of fake (close to 0)
        fp_confidences = probs[false_positives] if len(false_positives) > 0 else []
        fn_confidences = 1 - probs[false_negatives] if len(false_negatives) > 0 else []
        
        if len(fp_confidences) > 0:
            print(f"\n   Most confident False Positives (Real → Fake):")
            top_fp = np.argsort(fp_confidences)[-min(5, len(fp_confidences)):][::-1]
            for idx in top_fp:
                img_path = self.test_dataset.samples[false_positives[idx]]
                print(f"      {img_path.name} (confidence: {fp_confidences[idx]:.4f})")
        
        if len(fn_confidences) > 0:
            print(f"\n   Most confident False Negatives (Fake → Real):")
            top_fn = np.argsort(fn_confidences)[-min(5, len(fn_confidences)):][::-1]
            for idx in top_fn:
                img_path = self.test_dataset.samples[false_negatives[idx]]
                print(f"      {img_path.name} (confidence: {fn_confidences[idx]:.4f})")
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*60)
        print("        STARTING MODEL EVALUATION")
        print("="*60)
        
        # Get predictions
        labels, preds, probs = self.predict()
        
        # Calculate metrics
        metrics = self.calculate_metrics(labels, preds, probs)
        
        # Print results
        self.print_results(metrics)
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        self.plot_confusion_matrix(labels, preds)
        roc_auc = self.plot_roc_curve(labels, probs)
        self.plot_prediction_distribution(labels, probs)
        
        # Generate reports
        self.generate_classification_report(labels, preds)
        
        # Analyze errors
        self.analyze_errors(labels, preds, probs)
        
        # Save results
        self.save_results(metrics, roc_auc)
        
        print("\n" + "="*60)
        print("        EVALUATION COMPLETE!")
        print(f"All results saved to: {self.save_dir}/")
        print("="*60 + "\n")


def main():
    # Configuration
    MODEL_PATH = "saved/best_model.pth"  # Path to your trained model
    DATA_DIR = "dataset"                  # Your dataset directory
    SAVE_DIR = "test_results"             # Where to save results
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Please train the model first using: python train.py")
        return
    
    # Check if test data exists
    test_dir = Path(DATA_DIR) / 'test'
    if not test_dir.exists():
        print(f"❌ Error: Test directory not found at {test_dir}")
        print("   Please create test/real and test/fake folders with images")
        return
    
    # Run evaluation
    tester = ModelTester(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        device=device,
        save_dir=SAVE_DIR
    )
    
    tester.run_full_evaluation()


if __name__ == "__main__":
    main()