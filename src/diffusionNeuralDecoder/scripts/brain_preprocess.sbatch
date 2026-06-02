#!/bin/bash
#SBATCH --job-name=brain_preprocess
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/brain_preprocess%j.out
#SBATCH --error=logs/brain_preprocess%j.err

eval "$(micromamba shell hook --shell bash)"
micromamba activate neuralDecoder_env

mkdir -p logs 

cd /home/afan2025/Masters/Deep_Learning_Systems/CSMC_35200_BraintoTextGroup/DiffNeuralDecoder

python -m diffusionNeuralDecoder.preprocess_brain