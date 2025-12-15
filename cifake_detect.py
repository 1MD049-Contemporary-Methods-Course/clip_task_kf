"""
CIFAKE dataset Detection using CLIP
Detects AI-generated images in CIFAR-10 style datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse

class CIFAKEDataset(Dataset):
    """Dataset for CIFAKE images"""
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir: Root directory containing 'real' and 'fake' folders
            split: 'train' or 'test'
            transform: Optional transform to apply
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        
        # Load images from real and fake directories
        self.samples = []
        
        real_dir = self.root_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 0))  # 0 = real
        
        fake_dir = self.root_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 1))  # 1 = fake
        
        print(f"Loaded {len(self.samples)} images for {split} split")
        print(f"  Real: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  Fake: {sum(1 for _, label in self.samples if label == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CLIPZeroShotDetector:
    """Zero-shot CLIP detector using text prompts"""
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        
        # Define prompts for real vs fake images
        self.prompts = {
            'real': [
                "a real photograph",
                "a natural photo",
                "an authentic image",
                "a genuine photograph",
                "a photo taken with a camera"
            ],
            'fake': [
                "an AI-generated image",
                "a synthetic image",
                "a computer-generated image",
                "an artificial image",
                "a fake generated photo"
            ]
        }
        
        self.text_features = self._encode_text_prompts()
    
    def _encode_text_prompts(self) -> torch.Tensor:
        """Encode all text prompts"""
        all_prompts = self.prompts['real'] + self.prompts['fake']
        text_tokens = clip.tokenize(all_prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def predict(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict if images are real or fake
        Returns: (predictions, probabilities)
        """
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity with all prompts
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            
            # Average across prompts for each class
            n_real = len(self.prompts['real'])
            real_prob = similarity[:, :n_real].mean(dim=1)
            fake_prob = similarity[:, n_real:].mean(dim=1)
            
            # Normalize
            total = real_prob + fake_prob
            fake_prob = fake_prob / total
            
            predictions = (fake_prob > 0.5).cpu().numpy()
            probabilities = fake_prob.cpu().numpy()
        
        return predictions, probabilities

class CLIPFineTuned(nn.Module):
    """Fine-tuned CLIP model for binary classification"""
    def __init__(self, model_name: str = "ViT-B/32", freeze_backbone: bool = False):
        super().__init__()
        self.clip_model, _ = clip.load(model_name, device="cpu")
        
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Add classification head
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
        with torch.no_grad() if self.training == False else torch.enable_grad():
            image_features = self.clip_model.encode_image(images)
        
        logits = self.classifier(image_features.float())
        return logits

class BaselineDetector(nn.Module):
    """Simple CNN baseline detector"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, device: str, model_name: str) -> Dict:
    """Train a model"""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    best_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for images, labels in train_bar:
            images, labels = images.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}.pt')
    
    return history

def evaluate_model(model: nn.Module, data_loader: DataLoader, 
                   criterion: nn.Module, device: str) -> Tuple[float, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def comprehensive_evaluation(model, data_loader, device: str, model_name: str, 
                            is_zero_shot: bool = False) -> Dict:
    """Comprehensive model evaluation"""
    if not is_zero_shot:
        model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f'Evaluating {model_name}'):
            images = images.to(device)
            
            if is_zero_shot:
                preds, probs = model.predict(images)
                all_preds.extend(preds)
                all_probs.extend(probs)
            else:
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = probs > 0.5
                all_preds.extend(preds)
                all_probs.extend(probs)
            
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='CIFAKE Detection with CLIP')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to CIFAKE dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', 
                       choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='CLIP model variant')
    parser.add_argument('--skip_zero_shot', action='store_true', help='Skip zero-shot evaluation')
    parser.add_argument('--skip_fine_tune', action='store_true', help='Skip fine-tuning')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline training')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load CLIP preprocessing
    _, clip_preprocess = clip.load(args.clip_model, device=device)
    
    # Simple preprocessing for baseline
    simple_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets
    train_dataset = CIFAKEDataset(args.data_dir, 'train', transform=clip_preprocess)
    test_dataset = CIFAKEDataset(args.data_dir, 'test', transform=clip_preprocess)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    results = {}
    
    # 1. Zero-shot CLIP
    if not args.skip_zero_shot:
        print("\n" + "="*50)
        print("Zero-Shot CLIP Evaluation")
        print("="*50)
        zero_shot_detector = CLIPZeroShotDetector(args.clip_model, device)
        results['zero_shot_clip'] = comprehensive_evaluation(
            zero_shot_detector, test_loader, device, 'Zero-Shot CLIP', is_zero_shot=True
        )
    
    # 2. Fine-tuned CLIP
    if not args.skip_fine_tune:
        print("\n" + "="*50)
        print("Fine-Tuning CLIP")
        print("="*50)
        finetuned_model = CLIPFineTuned(args.clip_model, freeze_backbone=True)
        train_model(finetuned_model, train_loader, test_loader, 
                   args.epochs, args.lr, device, 'finetuned_clip')
        
        finetuned_model.load_state_dict(torch.load('best_finetuned_clip.pt'))
        results['finetuned_clip'] = comprehensive_evaluation(
            finetuned_model, test_loader, device, 'Fine-Tuned CLIP'
        )
    
    # 3. Baseline CNN
    if not args.skip_baseline:
        print("\n" + "="*50)
        print("Training Baseline CNN")
        print("="*50)
        baseline_train = CIFAKEDataset(args.data_dir, 'train', transform=simple_transform)
        baseline_test = CIFAKEDataset(args.data_dir, 'test', transform=simple_transform)
        baseline_train_loader = DataLoader(baseline_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        baseline_test_loader = DataLoader(baseline_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        baseline_model = BaselineDetector()
        train_model(baseline_model, baseline_train_loader, baseline_test_loader,
                   args.epochs, args.lr, device, 'baseline_cnn')
        
        baseline_model.load_state_dict(torch.load('best_baseline_cnn.pt'))
        results['baseline_cnn'] = comprehensive_evaluation(
            baseline_model, baseline_test_loader, device, 'Baseline CNN'
        )
    
    # Save all results
    with open('cifake_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Comparison plot
    if len(results) > 0:
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, (name, result) in enumerate(results.items()):
            values = [result[m] for m in metrics]
            plt.bar(x + i*width, values, width, label=name.replace('_', ' ').title())
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison on CIFAKE Dataset')
        plt.xticks(x + width, metrics)
        plt.legend()
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        
        print("\n" + "="*50)
        print("All results saved!")
        print("="*50)

if __name__ == '__main__':
    main()
