#!/bin/bash
#SBATCH --job-name=phoneme-dit-pretrain
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretrain%j.out
#SBATCH --error=logs/pretrain%j.err

eval "$(micromamba shell hook --shell bash)"
micromamba activate neuralDecoder_env

mkdir -p logs

cd /home/afan2025/Masters/Deep_Learning_Systems/CSMC_35200_BraintoTextGroup/DiffNeuralDecoder/diffusionNeuralDecoder

python scripts/pretrain.py \
    --epochs 200 \
    --batch-size 256 \
    --ckpt-every 5 \
    --log-every 100 \
    --global-seed 42