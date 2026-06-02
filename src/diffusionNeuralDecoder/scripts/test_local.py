# test_local.py
import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from diffusion_model import PhonemeDiT
from diffusion import create_diffusion
from diffusionNeuralDecoder.datasets import PhonemeDataset

device = torch.device("cpu")
model = PhonemeDiT(d_model=64, vocab_size=24, depth=2, max_len=96,
                    num_heads=4, mlp_ratio=4.0, use_cross_attention=False).to(device)

diffusion = create_diffusion(timestep_respacing="", noise_schedule="linear",
                              learn_sigma=False, sigma_small=True, predict_xstart=False)

# Fake batch
token_ids = torch.randint(0, 24, (4, 30))
mask = torch.ones(4, 30, dtype=torch.bool)
x_clean = model.embed_tok(token_ids)
t = torch.randint(0, 1000, (4,))
noise = torch.randn_like(x_clean)
x_noisy = diffusion.q_sample(x_clean, t, noise=noise)

out = model(x_noisy, mask, t, None, None)
loss = ((out - noise) ** 2).mean(dim=-1)
loss = (loss * mask.float()).sum() / mask.float().sum()
loss.backward()

print(f"Output shape: {out.shape}")
print(f"Loss: {loss.item():.4f}")
print("Forward + backward pass works.")