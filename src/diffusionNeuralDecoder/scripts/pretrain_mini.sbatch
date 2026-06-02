#!/bin/bash
#SBATCH --job-name=phoneme-dit-pretrain
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=30:00
#SBATCH --output=logs/pretrain_mini%j.out
#SBATCH --error=logs/pretrain_mini%j.err

eval "$(micromamba shell hook --shell bash)"
micromamba activate neuralDecoder_env

mkdir -p logs

cd /home/afan2025/Masters/Deep_Learning_Systems/CSMC_35200_BraintoTextGroup/DiffNeuralDecoder/diffusionNeuralDecoder

CUDA_LAUNCH_BLOCKING=1 python scripts/pretrain.py \
    --epochs 4 \
    --batch-size 4 \
    --ckpt-every 1 \
    --log-every 1 \