"""
Evaluate saved CIFAKE models on a new dataset
Simple script to test the trained models on different data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleImageDataset(Dataset):
    """Load images from fake-v2 and real folders"""
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load fake images
        fake_dir = self.data_dir / 'fake-v2'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 1))  # 1 = fake
        
        # Load real images
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 0))  # 0 = real
        
        print(f"Loaded {len(self.samples)} images")
        print(f"  Real: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  Fake: {sum(1 for _, label in self.samples if label == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if there's an error
            if self.transform:
                blank = Image.new('RGB', (224, 224), (0, 0, 0))
                return self.transform(blank), label
            return None, label

class CLIPFineTuned(nn.Module):
    """Same architecture as training script"""
    def __init__(self, model_name: str = "ViT-B/32"):
        super().__init__()
        self.clip_model, _ = clip.load(model_name, device="cpu")
        
        # Same classification head as training
        embed_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        logits = self.classifier(image_features.float())
        return logits

def evaluate_model(model, data_loader, device):
    """Evaluate and print results"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            if images is None:
                continue
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Handle single sample case
            if probs.ndim == 0:
                probs = np.array([probs])
            
            preds = probs > 0.5
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (Real correctly classified):  {cm[0,0]}")
    print(f"  False Positives (Real classified as Fake):   {cm[0,1]}")
    print(f"  False Negatives (Fake classified as Real):   {cm[1,0]}")
    print(f"  True Positives (Fake correctly classified):  {cm[1,1]}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix on New Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('new_dataset_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nSaved confusion matrix to: new_dataset_confusion_matrix.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluate saved model on new dataset')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to data folder containing fake-v2/ and real/')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model (.pt file)')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                       help='CLIP model variant used during training')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load CLIP preprocessing
    _, preprocess = clip.load(args.clip_model, device=device)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = SimpleImageDataset(args.data_dir, transform=preprocess)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = CLIPFineTuned(args.clip_model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Evaluate
    evaluate_model(model, data_loader, device)

if __name__ == '__main__':
    main()
