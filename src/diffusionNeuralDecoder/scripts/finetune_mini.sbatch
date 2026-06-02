#!/bin/bash
#SBATCH --job-name=phoneme-dit-finetune
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00
#SBATCH --output=logs/mini_finetune%j.out
#SBATCH --error=logs/mini_finetune%j.err

eval "$(micromamba shell hook --shell bash)"
micromamba activate neuralDecoder_env

mkdir -p logs

cd /home/afan2025/Masters/Deep_Learning_Systems/CSMC_35200_BraintoTextGroup/DiffNeuralDecoder/diffusionNeuralDecoder

python scripts/brain_finetune.py \
    --stage1-epochs 2 \
    --stage2-epochs 2 \
    --batch-size 4 \
    --ckpt-every 1 \
    --log-every 1 \