# CLIP Task: AI-Generated Image Detection

This repository contains implementations for detecting AI-generated images using CLIP (Contrastive Language-Image Pre-training) models. The project focuses on the CIFAKE dataset, which contains real and AI-generated CIFAR-10 style images.

## 📋 Overview

This project implements and compares three different approaches for detecting AI-generated images:

1. **Zero-Shot CLIP**: Uses pre-trained CLIP models with carefully crafted text prompts to classify images without any training
2. **Fine-Tuned CLIP**: Adds a classification head on top of CLIP's visual encoder and fine-tunes it on the CIFAKE dataset
3. **Baseline CNN**: A simple convolutional neural network trained from scratch for comparison

## ✨ Features

- Zero-shot detection using CLIP's vision-language understanding
- Fine-tuning capabilities with frozen or trainable CLIP backbone
- Baseline CNN implementation for performance comparison
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Confusion matrix visualization
- Model comparison plots
- Support for different CLIP model variants (ViT-B/32, ViT-B/16, ViT-L/14)
- Evaluation script for testing trained models on new datasets

## 🚀 Requirements

```
torch
torchvision
clip (OpenAI CLIP)
numpy
Pillow
tqdm
scikit-learn
matplotlib
seaborn
```

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/1MD049-Contemporary-Methods-Course/clip_task_kf.git
cd clip_task_kf
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install numpy pillow tqdm scikit-learn matplotlib seaborn
```

## 📁 Dataset Structure

The CIFAKE dataset should be organized as follows:

```
cifake/
├── train/
│   ├── real/
│   │   └── *.jpg
│   └── fake/
│       └── *.jpg
└── test/
    ├── real/
    │   └── *.jpg
    └── fake/
        └── *.jpg
```

For evaluation on new datasets (using `detect_superpotato.py`):
```
data/
├── real/
│   └── *.jpg
└── fake-v2/
    └── *.jpg
```

## 🎯 Usage

### Training and Evaluation

Run all three approaches on the CIFAKE dataset:

```bash
python cifake_detect.py --data_dir ./cifake --batch_size 64 --epochs 10 --lr 1e-4
```

#### Optional Arguments:
- `--clip_model`: Choose CLIP variant (`ViT-B/32`, `ViT-B/16`, `ViT-L/14`)
- `--skip_zero_shot`: Skip zero-shot evaluation
- `--skip_fine_tune`: Skip fine-tuning CLIP
- `--skip_baseline`: Skip baseline CNN training

### Evaluate on New Dataset

Use a trained model to evaluate on a new dataset:

```bash
python detect_superpotato.py --data_dir ./new_data --model_path best_finetuned_clip.pt --batch_size 32
```

### SLURM Cluster Usage

For running on UPPMAX (or similar HPC systems):

```bash
sbatch script.sh
```

## 🏗️ Model Architectures

### Zero-Shot CLIP
- Uses pre-trained CLIP visual encoder
- Text prompts for "real" and "fake" classes
- No training required
- Computes similarity between image and text embeddings

### Fine-Tuned CLIP
- Pre-trained CLIP visual encoder (frozen)
- Custom classification head:
  - Linear(embed_dim → 512) + ReLU + Dropout(0.3)
  - Linear(512 → 128) + ReLU + Dropout(0.3)
  - Linear(128 → 1)
- Binary classification with BCEWithLogitsLoss

### Baseline CNN
- 3 convolutional blocks (64→128→256 filters)
- MaxPooling after each block
- Fully connected layers (512 → 1)
- Trained from scratch on 32×32 images

## 📊 Results

Based on the CIFAKE test set evaluation:

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Zero-Shot CLIP | 50.20% | 50.10% | 99.81% | 66.71% | 0.589 |
| Fine-Tuned CLIP | **96.59%** | **96.15%** | **97.07%** | **96.61%** | **0.995** |
| Baseline CNN | 95.63% | 95.20% | 96.09% | 95.65% | 0.992 |

### Key Findings:
- **Fine-tuned CLIP** achieves the best overall performance across all metrics
- Zero-shot CLIP struggles without task-specific training
- Simple CNN baseline performs surprisingly well but slightly below fine-tuned CLIP
- Fine-tuned approach achieves near-perfect AUC (0.995)

## 📝 Scripts

### `cifake_detect.py`
Main training and evaluation script that:
- Implements all three model approaches
- Trains models on CIFAKE dataset
- Generates comprehensive evaluation metrics
- Creates visualization plots (confusion matrices, model comparison)
- Saves results to JSON and trained model weights

### `detect_superpotato.py`
Evaluation script for testing trained models on new datasets:
- Loads saved model weights
- Evaluates on custom dataset structure
- Generates confusion matrix and metrics
- Useful for testing model generalization

### Shell Scripts
- `script.sh`: SLURM job script for UPPMAX GPU cluster
- `script_snowy.sh`: Alternative SLURM script configuration
- `superpotato_script.sh`: SLURM script for evaluation on new dataset

## 📂 Output Files

After running, the following files are generated:

- `best_finetuned_clip.pt`: Best fine-tuned CLIP model weights
- `best_baseline_cnn.pt`: Best baseline CNN model weights
- `cifake_results.json`: Detailed results for all models
- `*_confusion_matrix.png`: Confusion matrices for each model
- `model_comparison.png`: Bar chart comparing all models
- `new_dataset_confusion_matrix.png`: Results on new dataset (if evaluated)

## 🎓 Academic Context

This project was developed as part of the 1MD049 Contemporary Methods course, demonstrating the application of modern deep learning techniques for synthetic image detection.

## 📄 License

This project is part of an academic course at Uppsala University.

## 🙏 Acknowledgments

- OpenAI CLIP: https://github.com/openai/CLIP
- CIFAKE Dataset for providing the benchmark data
- UPPMAX for computational resources