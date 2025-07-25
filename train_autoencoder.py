"""Training script for the MQGAN autoencoder.

This script loads all hyperparameters from a YAML configuration file and logs
training metrics to Weights & Biases.
"""
import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import yaml
import wandb

from preencoder import PreEncoder
from discriminators import MelSpectrogramPatchDiscriminator2D, MultiBinDiscriminator
from losses import LSGANLoss


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class RealMelSpectrogramDataset(Dataset):
    """Loads mel-spectrogram numpy files.

    Parameters
    ----------
    data_dir: str
        Directory containing ``*.npy`` files.
    crop_len: int | None
        If given, all sequences are cropped/padded to this length.
    """

    def __init__(self, data_dir: str, crop_len: int | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.crop_len = crop_len
        self.files = [
            Path(root) / f
            for root, _, files in os.walk(self.data_dir)
            for f in files
            if f.endswith(".npy")
        ]
        if not self.files:
            raise FileNotFoundError(f"No npy files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        mel = np.load(path)
        if mel.ndim != 2:
            raise ValueError(f"Invalid mel shape {mel.shape} in {path}")
        length = mel.shape[0]
        if self.crop_len is not None:
            target = self.crop_len
            if length > target:
                start = np.random.randint(0, length - target + 1)
                mel = mel[start : start + target]
            elif length < target:
                pad = np.zeros((target - length, mel.shape[1]), dtype=mel.dtype)
                mel = np.concatenate([mel, pad], axis=0)
            mel = mel[:target]
        return mel.astype(np.float32), min(length, self.crop_len or length)


def pad_collate_fn(batch):
    specs, lengths = zip(*batch)
    max_len = max(lengths)
    padded = []
    for spec in specs:
        if spec.shape[0] < max_len:
            pad = torch.zeros(max_len - spec.shape[0], spec.shape[1])
            spec = torch.cat([spec, pad], dim=0)
        padded.append(spec)
    return torch.stack(padded), torch.tensor(lengths)


def masked_l2_loss(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = sequence_mask(pred.shape[1], lengths).unsqueeze(1)
    diff = (pred - target) ** 2
    diff = diff.masked_fill(mask, 0.0)
    return diff.sum() / ( (~mask).sum() + 1e-8 )


def sequence_mask(max_length: int, lengths: torch.Tensor) -> torch.Tensor:
    range = torch.arange(max_length, device=lengths.device)
    return range.unsqueeze(0) >= lengths.unsqueeze(1)


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() and not config.get("no_cuda") else "cpu")
    random.seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))
    torch.manual_seed(config.get("seed", 42))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.get("seed", 42))

    dataset = RealMelSpectrogramDataset(config["data_dir"], config.get("crop_len"))
    val_split = config.get("validation_split", 0.1)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(
        train_ds,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        collate_fn=pad_collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=pad_collate_fn,
        pin_memory=device.type == "cuda",
    )

    gen_cfg = config.get("generator", {})
    model = PreEncoder(
        mel_channels=config.get("mel_channels", 80),
        channels=gen_cfg.get("channels", [192, 768, 1024, 1024]),
        kernel_sizes=gen_cfg.get("kernel_sizes", [3, 5, 7, 11]),
        dropout=gen_cfg.get("dropout", 0.0),
        fsq_levels=gen_cfg.get("fsq_levels", [8, 5, 5, 5]),
    ).to(device)

    disc_cfg = config.get("discriminator_patch", {})
    discriminator = MelSpectrogramPatchDiscriminator2D(
        config.get("mel_channels", 80),
        hidden_channels=disc_cfg.get("hidden_channels", [384, 384, 512, 512, 512]),
        kernel_sizes=disc_cfg.get("kernel_sizes", [7, 7, 5, 5, 3, 3]),
        stride=[tuple(s) for s in disc_cfg.get("stride", [(1, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])],
    ).to(device)

    disc_mb_cfg = config.get("discriminator_multibin", {})
    discriminator_mb = MultiBinDiscriminator(
        config.get("mel_channels", 80),
        n_bins=disc_mb_cfg.get("n_bins", 8),
        hidden_channels=disc_mb_cfg.get("hidden_channels", [128, 256, 256, 256, 256]),
        kernel_sizes=disc_mb_cfg.get("kernel_sizes", [7, 5, 5, 3, 3, 3]),
        n_no_strides=disc_mb_cfg.get("n_no_strides", 2),
    ).to(device)

    optimizer_g = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
    )
    optimizer_d = torch.optim.Adam(
        list(discriminator.parameters()) + list(discriminator_mb.parameters()),
        lr=config.get("lr", 1e-4) * 1.15,
        betas=(0.5, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer_g,
        lr_lambda=lambda step: min((step + 1) / config.get("warmup_steps", 3000), 1.0),
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    gan_loss = LSGANLoss()

    os.makedirs(config["output_dir"], exist_ok=True)
    wandb_run = None
    if config.get("wandb") and not config.get("wandb", {}).get("disable", False):
        wandb_run = wandb.init(
            project=config["wandb"].get("project", "mqgan"),
            entity=config["wandb"].get("entity"),
            name=config["wandb"].get("name"),
            config=config,
        )

    for epoch in range(1, config.get("num_epochs", 20) + 1):
        model.train()
        discriminator.train()
        discriminator_mb.train()
        total_loss = 0.0
        for real, lengths in train_dl:
            real = real.to(device)
            lengths = lengths.to(device)

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            with autocast(enabled=device.type == "cuda"):
                recon = model(real, lengths)
                loss_recon = masked_l2_loss(recon, real, lengths)

                real_logits, m_r = discriminator(real, lengths)
                fake_logits, m_f = discriminator(recon.detach(), lengths)
                loss_d = gan_loss.discriminator_loss(real_logits, fake_logits, m_r, m_f)

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)
            scaler.update()

            with autocast(enabled=device.type == "cuda"):
                gen_logits, m_g = discriminator(recon, lengths)
                loss_gan = gan_loss.generator_loss(gen_logits, m_g)
                loss = loss_recon + 15.0 * loss_gan

            scaler.scale(loss).backward()
            scaler.step(optimizer_g)
            scaler.update()
            scheduler.step()

            total_loss += loss_recon.item()
            if wandb_run:
                wandb.log({"loss/recon": loss_recon.item(), "loss/gan": loss_gan.item()})

        avg_recon = total_loss / len(train_dl)

        # ----- validation loop -----
        model.eval()
        val_recon = 0.0
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            for real, lengths in val_dl:
                real = real.to(device)
                lengths = lengths.to(device)
                recon = model(real, lengths)
                val_recon += masked_l2_loss(recon, real, lengths).item()
        val_recon /= max(len(val_dl), 1)
        model.train()

        print(f"Epoch {epoch}: recon loss {avg_recon:.4f}  val loss {val_recon:.4f}")

        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "loss/recon_epoch": avg_recon,
                "loss/val_recon": val_recon,
            })

        if epoch % config.get("save_interval", 1) == 0:
            path = Path(config["output_dir"]) / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_g.state_dict(),
            }, path)

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.setdefault("wandb", {})
        cfg["wandb"]["disable"] = True
    train(cfg)
