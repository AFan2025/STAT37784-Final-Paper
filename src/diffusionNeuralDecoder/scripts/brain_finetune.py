
# Libraries
import logging
import csv
import torch
import numpy as np
import argparse
import os
import sys
from collections import OrderedDict
from dotenv import load_dotenv
from copy import deepcopy
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader, random_split

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
from diffusionNeuralDecoder.datasets.speechDataset import BrainToTextDataset
from scripts.pretrain import (
    _get_env,
    _resolve_path,
    requires_grad,
    root_logger,
    save_checkpoint,
    training_step,
    update_ema,
)


def _configure_finetune_logger() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    root_logger.setLevel(logging.INFO)
    finetune_log = os.path.abspath(os.path.join(LOG_DIR, "finetune.log"))

    # Add exactly one finetune file handler even if this script is imported/run repeatedly.
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == finetune_log:
            return

    fh = logging.FileHandler(finetune_log)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(fh)


def _init_or_resume_metrics_file(metrics_path: str) -> None:
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    if os.path.exists(metrics_path):
        return
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "phase", "epoch", "step", "loss", "steps_per_sec", "lr"])


def _append_metric(metrics_path: str, stage: str, phase: str, epoch: int, step: int, loss: float,
                   steps_per_sec: str = "", lr: str = "") -> None:
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([stage, phase, epoch, step, loss, steps_per_sec, lr])


def _load_checkpoint_full(path: str, model: torch.nn.Module, ema: torch.nn.Module,
                          optimizer: torch.optim.Optimizer | None = None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    ema.load_state_dict(ckpt["ema"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def _prepare_step1_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    for name, param in model.named_parameters():
        if "brain_encoder" in name or "cross_attn" in name or "cross_attn_gate" in name or "ln_cross" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)


def _prepare_step2_optimizer(model: torch.nn.Module, unfreeze_top_n: int) -> torch.optim.Optimizer:
    for name, param in model.named_parameters():
        if "brain_encoder" in name or "cross_attn" in name or "cross_attn_gate" in name or "ln_cross" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    total_blocks = len(model.blocks)
    for i, block in enumerate(model.blocks):
        if i >= total_blocks - unfreeze_top_n:
            for param in block.parameters():
                param.requires_grad = True

    brain_params = [
        p
        for n, p in model.named_parameters()
        if ("brain_encoder" in n or "cross_attn" in n or "cross_attn_gate" in n or "ln_cross" in n)
        and p.requires_grad
    ]
    backbone_params = [
        p
        for n, p in model.named_parameters()
        if "blocks" in n and "cross_attn" not in n and "ln_cross" not in n and p.requires_grad
    ]

    logging.info("Step 2 trainable groups: brain=%d, backbone=%d", len(brain_params), len(backbone_params))
    return torch.optim.AdamW(
        [
            {"params": brain_params, "lr": 1e-4},
            {"params": backbone_params, "lr": 1e-5},
        ]
    )


def _batch_loss(model: torch.nn.Module, batch: dict, device: torch.device, diffusion_scheduler) -> torch.Tensor:
    x_ids = batch["phoneme_tokens"].to(device)
    x_mask = batch["phoneme_mask"].to(device)

    brain_z = batch["input_features"].to(device)
    brain_mask = batch["input_mask"].to(device)

    x = model.embed_tok(x_ids)
    t = torch.randint(0, diffusion_scheduler.num_timesteps, (x.shape[0],), device=device)
    return training_step(model, x, x_mask, t, diffusion_scheduler, brain_data=brain_z, brain_mask=brain_mask)


def _run_stage(stage_name: str,
               model: torch.nn.Module,
               ema: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               train_loader: DataLoader,
               val_loader: DataLoader,
               diffusion_scheduler,
               device: torch.device,
               metrics_path: str,
               ckpt_latest: str,
               ckpt_best: str,
               ckpt_done: str,
               start_epoch: int,
               num_epochs: int,
               train_steps: int,
               best_val_loss: float,
               log_every: int,
               ckpt_every: int) -> tuple[int, float]:
    logging.info("Starting %s from epoch %d/%d", stage_name, start_epoch, num_epochs)
    running_loss = 0.0
    log_steps = 0
    window_start = time()

    for epoch in tqdm(range(start_epoch, num_epochs), desc=f"{stage_name}"):
        model.train()

        for batch in train_loader:
            loss = _batch_loss(model, batch, device, diffusion_scheduler)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % log_every == 0:
                elapsed = max(time() - window_start, 1e-8)
                steps_per_sec = log_steps / elapsed
                avg_loss = running_loss / log_steps
                current_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    "[%s] (epoch=%04d step=%07d) train_loss=%.6f steps_per_sec=%.2f lr=%.6g",
                    stage_name,
                    epoch,
                    train_steps,
                    avg_loss,
                    steps_per_sec,
                    current_lr,
                )
                _append_metric(
                    metrics_path,
                    stage=stage_name,
                    phase="train",
                    epoch=epoch,
                    step=train_steps,
                    loss=float(avg_loss),
                    steps_per_sec=f"{steps_per_sec:.4f}",
                    lr=f"{current_lr:.8f}",
                )
                running_loss = 0.0
                log_steps = 0
                window_start = time()

        do_validation = (epoch % ckpt_every == 0) or (epoch == num_epochs - 1)
        if do_validation:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    val_loss = _batch_loss(model, batch, device, diffusion_scheduler)
                    val_losses.append(val_loss.item())

            avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                "[%s] (epoch=%04d step=%07d) val_loss=%.6f lr=%.6g",
                stage_name,
                epoch,
                train_steps,
                avg_val_loss,
                current_lr,
            )
            _append_metric(
                metrics_path,
                stage=stage_name,
                phase="val",
                epoch=epoch,
                step=train_steps,
                loss=avg_val_loss,
                steps_per_sec="",
                lr=f"{current_lr:.8f}",
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, ema, optimizer, epoch, train_steps, best_val_loss, ckpt_best)
                logging.info("[%s] New best checkpoint: %s", stage_name, ckpt_best)

            save_checkpoint(model, ema, optimizer, epoch, train_steps, best_val_loss, ckpt_latest)

    save_checkpoint(model, ema, optimizer, num_epochs - 1, train_steps, best_val_loss, ckpt_done)
    logging.info("[%s] Stage complete. Wrote done checkpoint: %s", stage_name, ckpt_done)
    return train_steps, best_val_loss


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


def main(args):
    _configure_finetune_logger()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    assert torch.cuda.is_available(), "Using a GPU"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD DATASET
    generator = torch.Generator().manual_seed(args.global_seed)
    dataset = BrainToTextDataset(data_path=PREPROCESSED_DATA_DIR, partition = 'train')
    train_size = int(len(dataset)*args.train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    logging.info("Dataset loaded from %s", PREPROCESSED_DATA_DIR)
    logging.info("Samples: total=%d train=%d val=%d", len(dataset), len(train_dataset), len(val_dataset))

    model = PhonemeDiT(
        d_model=D_MODEL,
        vocab_size=dataset.vocab_size,
        depth=MODEL_DEPTH,
        max_len=MAX_TEXT_LEN,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        use_cross_attention=False,  # now True
        z_brain_dim=Z_BRAIN_DIM,
        use_final_layer=False).to(device)

    pretrain_ckpt_path = _resolve_path(BASE_DIR, _get_env("PRETRAIN_CHECKPOINT", default=os.path.join(CHECKPOINT_DIR, "best.pt")))
    pretrained = torch.load(pretrain_ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(pretrained["model"], strict=False)
    logging.info("Loaded pretrain model from %s", pretrain_ckpt_path)
    logging.info("Missing keys (expected new finetune modules): %s", missing)
    logging.info("Unexpected keys: %s", unexpected)

    # EMA?
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)

    diffusion_scheduler = create_diffusion(
        timestep_respacing="",
        noise_schedule=DIFFUSION_NOISE_SCHEDULE,
        learn_sigma=False,
        sigma_small=True,
        predict_xstart=False,
    )
    logging.info("Diffusion scheduler created with noise schedule: %s", DIFFUSION_NOISE_SCHEDULE)


    run_id = int(time())
    metrics_path = os.path.join(LOG_DIR, f"finetune_loss_metrics_{run_id}.csv")
    _init_or_resume_metrics_file(metrics_path)
    logging.info("Streaming stage train/val metrics to %s", metrics_path)

    stage_paths = {
        "step1_latest": os.path.join(CHECKPOINT_DIR, "finetune_step1_latest.pt"),
        "step1_best": os.path.join(CHECKPOINT_DIR, "finetune_step1_best.pt"),
        "step1_done": os.path.join(CHECKPOINT_DIR, "finetune_step1_done.pt"),
        "step2_latest": os.path.join(CHECKPOINT_DIR, "finetune_step2_latest.pt"),
        "step2_best": os.path.join(CHECKPOINT_DIR, "finetune_step2_best.pt"),
        "step2_done": os.path.join(CHECKPOINT_DIR, "finetune_step2_done.pt"),
    }

    # if os.path.exists(stage_paths["step2_done"]):
    #     logging.info("Step 2 already complete (%s). Nothing to run.", stage_paths["step2_done"])
    #     return

    train_steps = 0
    unfreeze_top_n = 2

    # DEBUG INTERVENTION
    model_sd = model.state_dict()
    loaded_count = 0
    skipped_count = 0
    for key, val in pretrained['model'].items():
        if key in model_sd:
            if val.shape == model_sd[key].shape:
                loaded_count += 1
            else:
                skipped_count += 1
                logging.info(f"SHAPE MISMATCH: {key} pretrained={val.shape} model={model_sd[key].shape}")
        else:
            skipped_count += 1
            logging.info(f"KEY MISSING in model: {key}")

    logging.info(f"Loaded: {loaded_count}, Skipped: {skipped_count}")


    # Quick sanity check after loading
    logging.info("beginning randomized sanity check")
    model.eval()
    with torch.no_grad():
        fake_ids = torch.randint(0, 75, (4, 30), device=device)
        fake_mask = torch.ones(4, 30, dtype=torch.bool, device=device)
        fake_brain_tensor = torch.rand(4, 100, 2, 16, 8, device=device) * 2 - 1
        fake_brain_mask = torch.ones(4, 100, dtype=torch.bool, device=device)
        x = model.embed_tok(fake_ids)
        t = torch.randint(0, 1000, (4,), device=device)
        loss = training_step(
            model,
            x,
            fake_mask,
            t,
            diffusion_scheduler,
            brain_data=fake_brain_tensor,
            brain_mask=fake_brain_mask,
        )
        logging.info(f"Sanity check loss (should be ~0.001): {loss.item():.6f}")

    return # debugging statement


    # Step 2 resume path: skip step 1 entirely if a step 2 latest checkpoint exists.
    if os.path.exists(stage_paths["step2_latest"]):
        logging.info("Resuming directly in step2 from %s", stage_paths["step2_latest"])
        step2_opt = _prepare_step2_optimizer(model, unfreeze_top_n=unfreeze_top_n)
        ckpt = _load_checkpoint_full(stage_paths["step2_latest"], model, ema, step2_opt)
        start_epoch_2 = int(ckpt["epoch"]) + 1
        train_steps = int(ckpt.get("step", 0))
        best_val_2 = float(ckpt.get("val_loss", np.inf))

        _run_stage(
            stage_name="step2",
            model=model,
            ema=ema,
            optimizer=step2_opt,
            train_loader=train_loader,
            val_loader=val_loader,
            diffusion_scheduler=diffusion_scheduler,
            device=device,
            metrics_path=metrics_path,
            ckpt_latest=stage_paths["step2_latest"],
            ckpt_best=stage_paths["step2_best"],
            ckpt_done=stage_paths["step2_done"],
            start_epoch=start_epoch_2,
            num_epochs=args.stage2_epochs,
            train_steps=train_steps,
            best_val_loss=best_val_2,
            log_every=args.log_every,
            ckpt_every=args.ckpt_every,
        )
        logging.info("DONE")
        return

    # Step 1 section
    if os.path.exists(stage_paths["step1_done"]):
        logging.info("Step1 already done. Loading %s and continuing with step2.", stage_paths["step1_done"])
        ckpt = _load_checkpoint_full(stage_paths["step1_done"], model, ema, optimizer=None)
        train_steps = int(ckpt.get("step", 0))
    else:
        step1_opt = _prepare_step1_optimizer(model)
        if os.path.exists(stage_paths["step1_latest"]):
            logging.info("Resuming step1 from %s", stage_paths["step1_latest"])
            ckpt = _load_checkpoint_full(stage_paths["step1_latest"], model, ema, step1_opt)
            start_epoch_1 = int(ckpt["epoch"]) + 1
            train_steps = int(ckpt.get("step", 0))
            best_val_1 = float(ckpt.get("val_loss", np.inf))
        else:
            start_epoch_1 = 0
            best_val_1 = float("inf")

        if start_epoch_1 < args.stage1_epochs:
            train_steps, _ = _run_stage(
                stage_name="step1",
                model=model,
                ema=ema,
                optimizer=step1_opt,
                train_loader=train_loader,
                val_loader=val_loader,
                diffusion_scheduler=diffusion_scheduler,
                device=device,
                metrics_path=metrics_path,
                ckpt_latest=stage_paths["step1_latest"],
                ckpt_best=stage_paths["step1_best"],
                ckpt_done=stage_paths["step1_done"],
                start_epoch=start_epoch_1,
                num_epochs=args.stage1_epochs,
                train_steps=train_steps,
                best_val_loss=best_val_1,
                log_every=args.log_every,
                ckpt_every=args.ckpt_every,
            )
        else:
            save_checkpoint(
                model,
                ema,
                step1_opt,
                args.stage1_epochs - 1,
                train_steps,
                best_val_1,
                stage_paths["step1_done"],
            )

    # Step 2 section (always runs after step 1 completion unless already done/resumed above).
    step2_opt = _prepare_step2_optimizer(model, unfreeze_top_n=unfreeze_top_n)
    start_epoch_2 = 0
    best_val_2 = float("inf")
    if os.path.exists(stage_paths["step2_latest"]):
        ckpt = _load_checkpoint_full(stage_paths["step2_latest"], model, ema, step2_opt)
        start_epoch_2 = int(ckpt["epoch"]) + 1
        train_steps = int(ckpt.get("step", train_steps))
        best_val_2 = float(ckpt.get("val_loss", np.inf))

    if start_epoch_2 < args.stage2_epochs:
        _run_stage(
            stage_name="step2",
            model=model,
            ema=ema,
            optimizer=step2_opt,
            train_loader=train_loader,
            val_loader=val_loader,
            diffusion_scheduler=diffusion_scheduler,
            device=device,
            metrics_path=metrics_path,
            ckpt_latest=stage_paths["step2_latest"],
            ckpt_best=stage_paths["step2_best"],
            ckpt_done=stage_paths["step2_done"],
            start_epoch=start_epoch_2,
            num_epochs=args.stage2_epochs,
            train_steps=train_steps,
            best_val_loss=best_val_2,
            log_every=args.log_every,
            ckpt_every=args.ckpt_every,
        )

    logging.info("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5)
    args = parser.parse_args()
    main(args)