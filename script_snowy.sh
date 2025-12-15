#!/bin/bash
#SBATCH -A uppmax2025-2-346
#SBATCH -M snowy
#SBATCH -p node
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00

cd /proj/uppmax2025-2-346/clip_task_kf

source .venv/bin/activate

module load python_ML_packages/3.9.5-gpu

pip3 install git+https://github.com/openai/CLIP.git

python3 cifake_detect.py --data_dir ./cifake --batch_size 64 --epochs 10 --lr 1e-4
