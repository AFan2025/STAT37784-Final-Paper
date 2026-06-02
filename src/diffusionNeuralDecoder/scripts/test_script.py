
# Libraries
import logging
import argparse
import os
import sys
import editdistance

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../diffusionNeuralDecoder/scripts
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                      # .../diffusionNeuralDecoder
REPO_DIR = os.path.dirname(PROJECT_DIR)                        # .../DiffNeuralDecoder
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# Modules
from diffusion_model import PhonemeDiT
from diffusion import create_diffusion
from diffusionNeuralDecoder.datasets.speechDataset import BrainToTextDataset, ID_TO_PHONE
from scripts.pretrain import (
    _get_env,
    _resolve_path,
)
from scripts.brain_finetune import _batch_loss

def _configure_stdout_logger() -> logging.Logger:
    logger = logging.getLogger("test_script")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)
    return logger

BASE_DIR = _get_env("BASE_DIR", default=PROJECT_DIR)
COMPETITION_DATA_DIR = _resolve_path(BASE_DIR, _get_env("COMPETITION_DATA_DIR", default="../../../competition_data"))
PREPROCESSED_DATA_DIR = _get_env("PREPROCESSED_DATA_DIR", default="/net/scratch/afan2025/preprocessed_data")
CHECKPOINT_DIR = _resolve_path(BASE_DIR, _get_env("CHECKPOINT_DIR", default="./checkpoints"))

Z_BRAIN_DIM = _get_env("Z_BRAIN_DIM", int)
D_MODEL = _get_env("D_MODEL", int)
MAX_TEXT_LEN = _get_env("MAX_TEXT_LEN", int)
VOCAB_SIZE = _get_env("VOCAB_SIZE", int)
MODEL_DEPTH = _get_env("MODEL_DEPTH", int)
NUM_HEADS = _get_env("NUM_HEADS", int)
MLP_RATIO = _get_env("MLP_RATIO", float)
DECODER_METHOD = _get_env("DECODER_METHOD", default="nn")
DIFFUSION_NOISE_SCHEDULE = _get_env("DIFFUSION_NOISE_SCHEDULE", default="cosine")

@torch.no_grad()
def generate_from_brain(model, diffusion, batch, device):
    model.eval()

    brain_z = batch["input_features"].to(device)
    brain_mask = batch["input_mask"].to(device)
    x_real = batch["phoneme_tokens"].to(device)
    x_mask = batch["phoneme_mask"].to(device)

    B = brain_z.shape[0]
    
    # Start from pure noise
    x = torch.randn(B, MAX_TEXT_LEN, model.d_model, device=device)
    dummy_mask = torch.ones(B, MAX_TEXT_LEN, dtype=torch.bool, device=device)
    
    # # Encode brain signal once
    # brain_enc = model.brain_encoder(brain_z)
    # brain_global = brain_enc.mean(dim=1)
    
    # Full denoising loop
    for i in reversed(range(diffusion.num_timesteps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        
        # Model predicts noise
        noise_pred = model(x, dummy_mask, t, brain_z, brain_mask)
        
        # Recover x_0 estimate
        alpha_cumprod = diffusion.alphas_cumprod[i]
        x_start = (1.0 / np.sqrt(alpha_cumprod)) * x - \
              (np.sqrt(1.0 - alpha_cumprod) / np.sqrt(alpha_cumprod)) * noise_pred
        
        if i > 0:
            # Posterior step
            alpha_cumprod_prev = diffusion.alphas_cumprod_prev[i]
            beta = diffusion.betas[i]
            coef1 = beta * np.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            coef2 = (1.0 - alpha_cumprod_prev) * np.sqrt(1.0 - beta) / (1.0 - alpha_cumprod)
            mean = coef1 * x_start + coef2 * x
            posterior_var = beta * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            x = mean + np.sqrt(posterior_var) * torch.randn_like(x)
        else:
            x = x_start
    
    # Decode to tokens
    token_ids = model.decode_tok(x)

    batch_per = []
    for i in range(len(token_ids)):
        valid = x_mask[i].bool()
        seq_len = int(valid.sum().item())
        if seq_len == 0:
            return 0.0
        
        pred_ids = token_ids[i].tolist()[:seq_len]
        ref_ids = x_real[i].tolist()[:seq_len]
        dist = editdistance.eval(pred_ids, ref_ids)
        per = dist/seq_len
        batch_per.append(per)

    avg_edit_dist = sum(batch_per) / len(batch_per)

    return avg_edit_dist

def main(args):
    logger = _configure_stdout_logger()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    assert torch.cuda.is_available(), "Using a GPU"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)

    logger.info("Initializing dataset from partition %s", args.partition)
    dataset = BrainToTextDataset(data_path=PREPROCESSED_DATA_DIR, partition = args.partition)

    data_loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    logger.info("Dataset loaded from %s (%s)", PREPROCESSED_DATA_DIR, args.partition)
    logger.info("Samples: total=%d", len(dataset))

    model = PhonemeDiT(
        d_model=D_MODEL,
        vocab_size=dataset.vocab_size,
        depth=MODEL_DEPTH,
        max_len=MAX_TEXT_LEN,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        use_cross_attention=args.conditional == "conditional",  
        z_brain_dim=Z_BRAIN_DIM,
        use_final_layer=False,
        ).to(device)
    
    if args.checkpoint is not None:
        checkpoint_name = args.checkpoint
    elif args.conditional == "conditional":
        checkpoint_name = "finetune_step2_best.pt"
    else:
        checkpoint_name = "best.pt"
    ckpt_path = _resolve_path(BASE_DIR, os.path.join(CHECKPOINT_DIR, checkpoint_name))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_checkpoint = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(model_checkpoint["model"], strict=False)
    logger.info("Loaded model from %s", ckpt_path)
    logger.info("Missing keys after load: %s", missing)
    logger.info("Unexpected keys after load: %s", unexpected)

    # diffusion scheduler but for inference time
    diffusion_scheduler = create_diffusion(
        timestep_respacing="ddim50", #maybe should be different for test inference?
        noise_schedule=DIFFUSION_NOISE_SCHEDULE,
        learn_sigma=False,
        sigma_small=True,
        predict_xstart=False,
    )
    logger.info("Diffusion scheduler created with noise schedule: %s", DIFFUSION_NOISE_SCHEDULE)

    model.eval()

    if args.conditional == "conditional":
        logger.info("Running conditional eval with brain-conditioned batches")
        running = 0.0
        seen = 0
        with torch.no_grad():
            for step_idx, batch in enumerate(data_loader, start=1):
                if step_idx == 1:
                    logger.info(
                        "First batch shapes: input_features=%s, input_mask=%s, phoneme_tokens=%s, phoneme_mask=%s",
                        tuple(batch["input_features"].shape),
                        tuple(batch["input_mask"].shape),
                        tuple(batch["phoneme_tokens"].shape),
                        tuple(batch["phoneme_mask"].shape),
                    )
                # loss = _batch_loss(model, batch, device, diffusion_scheduler)
                per = generate_from_brain(model, diffusion_scheduler, batch, device)
                running += per
                seen += 1 

                if step_idx % args.log_every == 0:
                    logger.info("[conditional] step=%d avg_loss=%.6f", step_idx, running / seen)

        if seen == 0:
            logger.warning("No batches were processed. Check partition and batch-size/drop_last settings.")
        else:
            logger.info("[conditional] completed batches=%d final_avg_loss=%.6f", seen, running / seen)

    else:
        logger.info("Running unconditional sanity decode with random latent inputs")
        with torch.no_grad():
            for sample_idx in range(1, args.num_unconditional_samples + 1):

                # Use random hidden states as sanity inputs for the unconditional forward path.
                noise = torch.randn((1, args.unconditional_seq_len, D_MODEL), device=device)
                noise_mask = torch.ones(1, args.unconditional_seq_len, dtype=torch.bool, device=device)
                t = torch.randint(0, diffusion_scheduler.num_timesteps, (noise.shape[0],), device=device)

                # Unconditional model still uses the same forward signature; brain inputs are None.
                pred = model(noise, noise_mask, t, None, None)
                token_id_seq = model.decode_tok(pred)
                phoneme_seq = [ID_TO_PHONE[out] for out in token_id_seq[0].detach().cpu().tolist()]

                if sample_idx <= args.print_first_n_samples:
                    logger.info(
                        "[unconditional] sample=%d token_ids=%s sequence=%s",
                        sample_idx,
                        token_id_seq[0].detach().cpu().tolist(),
                        phoneme_seq,
                    )

                if sample_idx % args.log_every == 0:
                    logger.info("[unconditional] generated %d/%d samples", sample_idx, args.num_unconditional_samples)

        logger.info("[unconditional] completed %d samples", args.num_unconditional_samples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--conditional", type=str, choices=["conditional","unconditional"], default="conditional")
    parser.add_argument("--partition", type=str, choices=["test", "competitionHoldOut"], default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-unconditional-samples", type=int, default=100)
    parser.add_argument("--unconditional-seq-len", type=int, default=30)
    parser.add_argument("--print-first-n-samples", type=int, default=3)
    args = parser.parse_args()
    main(args)

# Run command:
# python scripts/test_script.py --conditional {conditional|unconditional} --partition {test|competitionHoldOut} --batch-size 128 --num-workers 4 --log-every 100 --global-seed 0 --checkpoint <optional_checkpoint_name.pt> --num-unconditional-samples 100 --unconditional-seq-len 30 --print-first-n-samples 3    