#!/bin/bash
#SBATCH -A uppmax2025-2-346
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -J cifake_test

module load Python/3.12.3-GCCcore-13.3.0

cd /proj/uppmax2025-2-346/clip_task_kf

source .venv/bin/activate

python3 cifake_detect.py --data_dir ./cifake --batch_size 64 --epochs 10 --lr 1e-4
