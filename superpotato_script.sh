#!/bin/bash
#SBATCH -A uppmax2025-2-346
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH -J superpotato

module load Python/3.12.3-GCCcore-13.3.0

cd /proj/uppmax2025-2-346/clip_task_kf

source .venv/bin/activate

python3 detect_superpotato.py --data_dir ./ai-recog-superpotato --model_path ./results/best_finetuned_clip.pt
