# train.py

import os
import yaml
import argparse
import numpy as np
import torch
import glob

# having this True is bad (both for CUDA and ROCm) when we use diff lens each batch
torch.backends.cudnn.benchmark = False

import wandb
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from functools import partial

from discriminators import MelSpectrogramPatchDiscriminator2D, MultiBinDiscriminator
from losses import LSGANLoss, MaskedMelLoss
from preencoder import PreEncoder as MVQGenerator


def get_param_num(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


# --- Utility Functions ---

def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(pred)
    diff = torch.abs(pred - target)
    diff = diff.masked_fill(mask, 0.0)
    valid_cnt = (~mask).sum()
    return diff.sum() / (valid_cnt + eps)


def plot_mel_spectrograms(spectrograms, titles, vmin, vmax, save_path=None, main_title='Mel Spectrograms'):
    """
    Plots a list of mel spectrograms stacked vertically.
    """
    num_specs = len(spectrograms)
    fig, axes = plt.subplots(num_specs, 1, figsize=(10, 4 * num_specs))
    if num_specs == 1:
        axes = [axes]

    for i in range(num_specs):
        ax = axes[i]
        spec = spectrograms[i]
        if isinstance(spec, torch.Tensor):
            spec = spec.float().cpu().detach().numpy()
        if spec.ndim != 2:
            print(f"Error plotting: Spectrogram has unexpected shape {spec.shape}. Expected 2D.")
            continue

        spectrogram_display = np.transpose(spec, (1, 0))
        im = ax.imshow(spectrogram_display, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='magma')
        fig.colorbar(im, ax=ax, format='%+2.0f')
        ax.set_title(titles[i])
        ax.set_ylabel('Frequency')

    axes[-1].set_xlabel('Time')
    plt.suptitle(main_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    img = wandb.Image(plt)
    plt.close(fig)
    return img


# --- Dataset ---

class RealMelSpectrogramDataset(Dataset):
    def __init__(self, data_dir: str, crop_len=None):
        self.real_dir = data_dir
        self.crop_len = crop_len
        print(f"Crop len: {self.crop_len}")
        if not os.path.isdir(self.real_dir):
            raise FileNotFoundError(f"Directory not found: {self.real_dir}")
        self.filenames = [
            os.path.join(root, fn)
            for root, _, files in os.walk(self.real_dir)
            for fn in files if fn.endswith(".npy")
        ]
        # self.filenames = self.filenames[:2000]
        if not self.filenames:
            print(f"Warning: No .npy files found in {self.real_dir} (recursively).")
        else:
            print(f"Found {len(self.filenames)} .npy files.")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]
        try:
            mel = np.load(path)
        except Exception as e:
            print(f"[Load error] {path}: {e}")
            return None

        if mel.ndim != 2:
            print(f"[Shape error] {path}: shape={mel.shape}")
            return None

        full_len = mel.shape[0]
        target = self.crop_len

        if target is not None:
            if full_len > target:
                start = np.random.randint(0, full_len - target + 1)
                mel = mel[start: start + target]
            elif full_len < target:
                pad = np.zeros((target - full_len, mel.shape[1]), dtype=mel.dtype)
                mel = np.concatenate([mel, pad], axis=0)
            mel = mel[: target]
            assert mel.shape[0] == target, f"Unexpected len {mel.shape[0]} vs {target}"

        mel_len = min(full_len, target) if target is not None else full_len
        mel = mel.astype(np.float32)
        mel = torch.as_tensor(mel, dtype=torch.float32)
        return mel, int(mel_len), os.path.basename(path)


# ────────────────────────────────────────────────────────────────
def pad_collate_fn(
        batch,
        crop_lens=None  # e.g. 256  *or*  [128,192,256]
):
    """
    • If crop_lens is None          ➜ original behaviour (pad to max‐length).
    • If crop_lens is an int        ➜ always crop/pad to that length.
    • If crop_lens is a list/tuple  ➜ pick one length at random *per batch*.
    """
    # ── filter out failed loads ─────────────────────────────────────────────
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None

    real_spectrograms, mel_lens, filenames = zip(*batch)

    # ── decide the target length for this batch ────────────────────────────
    tgt_len = None
    if crop_lens is not None:
        if isinstance(crop_lens, (list, tuple)):
            tgt_len = int(random.choice(crop_lens))
        else:  # single int
            tgt_len = int(crop_lens)

    # ── fast-path: already equal & no explicit target ──────────────────────
    if tgt_len is None and len({m.shape[0] for m in real_spectrograms}) == 1:
        real_padded_stacked = torch.stack(
            [torch.as_tensor(m, dtype=torch.float32) for m in real_spectrograms]
        )
        mel_lens_tensor = torch.tensor(mel_lens, dtype=torch.int32)
        return real_padded_stacked, mel_lens_tensor, filenames

    # ── crop / pad each item to tgt_len (or max-len when tgt_len is None) ──
    if tgt_len is None:
        tgt_len = max(mel_lens)  # original behaviour

    real_padded, new_lens = [], []
    for mel, full_len in zip(real_spectrograms, mel_lens):
        if full_len > tgt_len:  # random crop
            start = random.randint(0, full_len - tgt_len)
            mel = mel[start: start + tgt_len]
        elif full_len < tgt_len:  # right-pad with zeros
            pad_amt = tgt_len - full_len
            mel = F.pad(
                torch.as_tensor(mel, dtype=torch.float32),
                (0, 0, 0, pad_amt),
                mode="constant",
                value=0,
            )
        else:  # already tgt_len
            mel = torch.as_tensor(mel, dtype=torch.float32)

        real_padded.append(mel)
        new_lens.append(min(full_len, tgt_len))

    real_padded_stacked = torch.stack(real_padded)
    mel_lens_tensor = torch.tensor(new_lens, dtype=torch.int32)
    return real_padded_stacked, mel_lens_tensor, filenames


# --- Trainer Class ---

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.start_epoch = 1
        self._setup_seeds()

        self.train_loader, self.eval_dataset = self._get_dataloaders()

        self.generator, self.patch_discriminator, self.multibin_discriminator = self._init_models()

        self.optimizer_g, self.optimizer_d = self._init_optimizers()
        self.scheduler_g = self._init_schedulers()

        self.gan_loss = LSGANLoss().to(self.device)
        self.recon_loss_all = MaskedMelLoss("mse").to(self.device)
        self.recon_loss_group = MaskedMelLoss("mse", group_size=16).to(self.device)

        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        print(f"Using bfloat16: {self.use_bf16}")

        self.scaler_g = GradScaler(enabled=self.device.type == 'cuda' and not self.use_bf16)
        self.scaler_d = GradScaler(enabled=self.device.type == 'cuda' and not self.use_bf16)

        self._init_checkpoint_handling()
        self._init_wandb()

    def _get_device(self):
        use_cuda = torch.cuda.is_available() and not self.config['training']['no_cuda']
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f"Using device: {device}")
        return device

    def _setup_seeds(self):
        seed = self.config['training']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    def _get_dataloaders(self):
        print("Loading dataset...")
        try:
            full_dataset = RealMelSpectrogramDataset(self.config['data']['data_dir'], None)
            if len(full_dataset) == 0:
                raise ValueError("Dataset is empty after initialization.")

            eval_size = int(self.config['data']['validation_split'] * len(full_dataset))
            train_size = len(full_dataset) - eval_size
            if train_size <= 0 or eval_size < 0:
                raise ValueError(f"Invalid train/eval split sizes. Train: {train_size}, Eval: {eval_size}.")

            print(f"Dataset size: {len(full_dataset)}. Splitting into {train_size} train and {eval_size} eval samples.")
            generator = torch.Generator().manual_seed(self.config['training']['seed'])
            train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)

            # pick one crop size per batch
            collate = partial(pad_collate_fn, crop_lens=self.config['data']['crop_len'])

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['data']['batch_size'],
                shuffle=True,
                num_workers=self.config['data']['num_workers'],
                collate_fn=collate,
                pin_memory=self.device.type == 'cuda'
            )
            return train_loader, eval_dataset
        except (FileNotFoundError, ValueError) as e:
            print(f"Error initializing dataset: {e}")
            exit()

    def _init_models(self):
        print("Initializing models...")
        gen_cfg = self.config['model']['generator']
        generator = MVQGenerator(
            mel_channels=self.config['model']['mel_channels'],
            channels=gen_cfg['channels'],
            kernel_sizes=gen_cfg['kernel_sizes'],
            dropout=gen_cfg['dropout'],
            fsq_levels=gen_cfg['fsq_levels']
        ).to(self.device)

        patch_cfg = self.config['model']['discriminator_patch']
        patch_discriminator = MelSpectrogramPatchDiscriminator2D(
            self.config['model']['mel_channels'],
            hidden_channels=patch_cfg['hidden_channels'],
            kernel_sizes=patch_cfg['kernel_sizes'],
            stride=patch_cfg['strides']
        ).to(self.device)

        mb_cfg = self.config['model']['discriminator_multibin']
        multibin_discriminator = MultiBinDiscriminator(
            self.config['model']['mel_channels'],
            hidden_channels=mb_cfg['hidden_channels'],
            kernel_sizes=mb_cfg['kernel_sizes'],
            n_bins=mb_cfg['n_bins'],
            n_no_strides=mb_cfg['n_no_strides']
        ).to(self.device)

        print(f"Number of Generator Parameters: {get_param_num(generator) / 1e6:.2f}M")
        print(f"Number of Patch Discriminator Parameters: {get_param_num(patch_discriminator) / 1e6:.2f}M")
        print(f"Number of Multi-bin Discriminator Parameters: {get_param_num(multibin_discriminator) / 1e6:.2f}M")

        return generator, patch_discriminator, multibin_discriminator

    def _init_optimizers(self):
        train_cfg = self.config['training']
        optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=train_cfg['lr'],
            betas=(train_cfg['beta1'], train_cfg['beta2'])
        )
        optimizer_d = optim.Adam(
            list(self.patch_discriminator.parameters()) + list(self.multibin_discriminator.parameters()),
            lr=train_cfg['lr'] * train_cfg['lr_d_factor'],
            betas=(train_cfg['d_beta1'], train_cfg['d_beta2'])
        )
        return optimizer_g, optimizer_d

    def _init_schedulers(self):
        lr_lambda = lambda step: min((step + 1) / self.config['training']['warmup_steps'], 1.0)
        scheduler_g = LambdaLR(self.optimizer_g, lr_lambda)
        return scheduler_g

    def _init_wandb(self):
        wandb.init(
            project=self.config['project_name'],
            entity=self.config['logging']['wandb']['entity'],
            config=self.config
        )
        wandb.watch(self.generator, log="all")

    def _init_checkpoint_handling(self):
        output_dir = self.config['data']['output_dir']
        latest_ckpt = max(glob.glob(os.path.join(output_dir, 'checkpoint_epoch_*.pth')), key=os.path.getctime, default=None)

        if latest_ckpt:
            print(f"Found latest checkpoint: {latest_ckpt}")
            self._load_full_checkpoint(latest_ckpt)
        else:
            self._load_pretrained_generator()

    def _load_full_checkpoint(self, ckpt_path):
        print(f"=> Loading full checkpoint for resuming training from '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
        self.scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}. Resuming training from epoch {self.start_epoch}.")

    def _load_pretrained_generator(self):
        ckpt_path = self.config['training']['pretrained']
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"=> Loading pretrained generator from '{ckpt_path}'")
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.generator.load_state_dict(clean_state_dict, strict=False)
            print(f"Loaded pretrained generator checkpoint.")
        else:
            print("No pretrained checkpoint specified or found. Training from scratch.")

    def _train_discriminator(self, real_spectrograms, recon_spectrograms, mel_lens):
        self.optimizer_d.zero_grad()

        real_logits, real_mask, _ = self.patch_discriminator(real_spectrograms, mel_lens, return_features=True)
        fake_logits, fake_mask = self.patch_discriminator(recon_spectrograms.detach(), mel_lens)
        loss_d1 = self.gan_loss.discriminator_loss(real_logits, fake_logits, real_mask, fake_mask)

        real_logits_l2, real_mask_l2, _ = self.multibin_discriminator(real_spectrograms, mel_lens, return_features=True)
        fake_logits_l2, fake_mask_l2 = self.multibin_discriminator(recon_spectrograms.detach(), mel_lens)

        loss_mbd = torch.tensor(0.0, device=self.device)
        for i, r_logits in enumerate(real_logits_l2):
            d_loss_mask_r = real_mask_l2[0]
            d_loss_mask_f = fake_mask_l2[0]
            f_logits = fake_logits_l2[i]
            loss_mbd += self.gan_loss.discriminator_loss(r_logits, f_logits, d_loss_mask_r, d_loss_mask_f)
        if len(real_logits_l2) > 0:
            loss_mbd /= len(real_logits_l2)

        loss_d = loss_d1 + loss_mbd
        self.scaler_d.scale(loss_d).backward()
        clip_value = self.config['training'].get('clip_grad_norm', 1.0)
        if clip_value:
            torch.nn.utils.clip_grad_norm_(
                list(self.patch_discriminator.parameters()) + list(self.multibin_discriminator.parameters()),
                clip_value
            )
        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()

        return loss_d.item()

    def _train_generator(self, real_spectrograms, recon_pre, recon_post, mel_lens):
        self.optimizer_g.zero_grad()

        self.patch_discriminator.eval()
        self.multibin_discriminator.eval()

        # Reconstruction losses
        loss_recon_pre_all = self.recon_loss_all(recon_pre, real_spectrograms, mel_lens)
        loss_recon_pre_g = self.recon_loss_group(recon_pre, real_spectrograms, mel_lens)
        loss_recon_pre = loss_recon_pre_all + loss_recon_pre_g * 0.25

        loss_recon_post_all = self.recon_loss_all(recon_post, real_spectrograms, mel_lens)
        loss_recon_post_g = self.recon_loss_group(recon_post, real_spectrograms, mel_lens)
        loss_recon_post = loss_recon_post_all + loss_recon_post_g * 0.25

        if self.epoch >= self.config['training']['discriminator_train_start_epoch']:
            gen_logits, gen_mask, gen_feats = self.patch_discriminator(recon_post, mel_lens,
                                                                        return_features=True)
            gen_logits_l2, gen_mask_l2, gen_feats_l2 = self.multibin_discriminator(recon_post, mel_lens,
                                                                                    return_features=True)

            loss_gan_d1 = self.gan_loss.generator_loss(gen_logits, gen_mask)

            loss_gan_mbd = torch.tensor(0.0, device=self.device)
            for i, g_logits in enumerate(gen_logits_l2):
                g_loss_mask = gen_mask_l2[0]
                loss_gan_mbd += self.gan_loss.generator_loss(g_logits, g_loss_mask)
            if len(gen_logits_l2) > 0:
                loss_gan_mbd /= len(gen_logits_l2)

            loss_gan = 0.5 * (loss_gan_d1 + loss_gan_mbd)

            c_weights = self.config['training']['loss_weights']
            current_gloss_lambda = c_weights['Gloss_lambda']
            current_fmloss_lambda = c_weights['fm_lambda']
        else:
            loss_gan = torch.tensor(0.0, device=self.device)
            current_gloss_lambda = 0.0
            current_fmloss_lambda = 0.0

        loss_fm = torch.tensor(0.0, device=self.device)
        if self.config['training']['use_fm_loss'] and self.epoch >= self.config['training'][
            'discriminator_train_start_epoch']:
            with torch.no_grad():
                _, _, real_feats = self.patch_discriminator(real_spectrograms, mel_lens, return_features=True)
                _, _, real_feats_l2 = self.multibin_discriminator(real_spectrograms, mel_lens, return_features=True)

            loss_fm_d1 = torch.tensor(0.0, device=self.device)
            for (rf, mask), (ff, _) in zip(real_feats, gen_feats):
                loss_fm_d1 += masked_mae(ff, rf, mask)
            if len(real_feats) > 0:
                loss_fm_d1 /= len(real_feats)

            loss_fm_mbd = torch.tensor(0.0, device=self.device)
            for i in range(len(gen_feats_l2)):
                r_feats = real_feats_l2[i]
                g_feats = gen_feats_l2[i]
                for (rf, mask), (ff, _) in zip(r_feats, g_feats):
                    loss_fm_mbd += masked_mae(ff, rf, mask)
                if len(r_feats) > 0:
                    loss_fm_mbd /= len(r_feats)
            if len(gen_feats_l2) > 0:
                loss_fm_mbd /= len(gen_feats_l2)

            loss_fm = 0.5 * (loss_fm_d1 + loss_fm_mbd)

        weights = self.config['training']['loss_weights']
        total_loss_g = (loss_recon_pre * weights.get('recon_lambda_pre', 1.0) +
                        loss_recon_post * weights.get('recon_lambda_post', 2.0) +
                        loss_gan * current_gloss_lambda +
                        loss_fm * current_fmloss_lambda)

        self.scaler_g.scale(total_loss_g).backward()
        clip_value = self.config['training'].get('clip_grad_norm', 1.0)
        if clip_value:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                clip_value
            )
        self.scaler_g.step(self.optimizer_g)
        self.scheduler_g.step()
        self.scaler_g.update()

        return {
            "loss_g_total": total_loss_g.item(),
            "loss_recon_pre": loss_recon_pre.item(),
            "loss_recon_post": loss_recon_post.item(),
            "loss_gan": loss_gan.item(),
            "loss_fm": loss_fm.item()
        }

    def _train_epoch(self, epoch):
        self.generator.train()
        self.patch_discriminator.train()
        self.multibin_discriminator.train()
        self.epoch = epoch

        loop = tqdm(self.train_loader, leave=True, desc=f"Epoch [{epoch}/{self.config['training']['num_epochs']}]")

        for batch_idx, batch_data in enumerate(loop):
            if batch_data[0] is None:
                print(f"Skipping empty batch {batch_idx + 1}")
                continue

            real_spectrograms, mel_lens, filenames = batch_data
            real_spectrograms = real_spectrograms.to(self.device)
            mel_lens = mel_lens.to(self.device)

            if real_spectrograms.size(0) == 0:
                continue

            with autocast(enabled=self.device.type == 'cuda', dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
                recon_pre, recon_post = self.generator(real_spectrograms, mel_lens)

                loss_d = 0.0
                if epoch >= self.config['training']['discriminator_train_start_epoch']:
                    loss_d = self._train_discriminator(real_spectrograms, recon_post, mel_lens)

                g_losses = self._train_generator(real_spectrograms, recon_pre, recon_post, mel_lens)

            loop.set_postfix(D_loss=loss_d, G_loss=g_losses['loss_g_total'], Recon_Post=g_losses['loss_recon_post'])
            wandb.log({
                "loss_d": loss_d,
                **g_losses,
                "learning_rate": self.scheduler_g.get_last_lr()[0]
            })

        self._log_train_images(epoch, real_spectrograms, recon_pre, recon_post, mel_lens, filenames)

    def _log_train_images(self, epoch, real_spectrograms, recon_pre, recon_post, mel_lens, filenames):
        if not (real_spectrograms is not None and recon_pre is not None and recon_post is not None):
            return

        plot_cfg = self.config['logging']
        plot_dir = os.path.join(self.config['data']['output_dir'], 'plots')

        with torch.no_grad():
            vmin = min(real_spectrograms.min().item(), recon_pre.min().item(), recon_post.min().item())
            vmax = max(real_spectrograms.max().item(), recon_pre.max().item(), recon_post.max().item())

        num_plots = min(plot_cfg['num_plot_examples'], real_spectrograms.size(0))
        log_dict = {}
        for i in range(num_plots):
            actual_len = mel_lens[i].item()
            original_spec = real_spectrograms[i, :actual_len, :]
            recon_spec_pre = recon_pre[i, :actual_len, :]
            recon_spec_post = recon_post[i, :actual_len, :]
            filename = filenames[i] if i < len(filenames) else f"Unknown_{i}"

            save_path_orig = os.path.join(plot_dir,
                                          f'epoch_{epoch:03d}_train_orig_{i + 1}_{os.path.splitext(filename)[0]}.png')
            save_path_recon_pre = os.path.join(plot_dir,
                                               f'epoch_{epoch:03d}_train_recon_pre_{i + 1}_{os.path.splitext(filename)[0]}.png')
            save_path_recon_post = os.path.join(plot_dir,
                                                f'epoch_{epoch:03d}_train_recon_post_{i + 1}_{os.path.splitext(filename)[0]}.png')

            log_dict[f"train_comparison_{i + 1}"] = plot_mel_spectrograms(
                [original_spec, recon_spec_pre, recon_spec_post],
                ['Original', 'Reconstructed (Pre-Refiner)', 'Reconstructed (Post-Refiner)'],
                vmin, vmax, save_path_orig, f'Epoch {epoch} Train - {os.path.splitext(filename)[0]}'
            )
        wandb.log(log_dict)

    def _evaluate(self, epoch):
        self.generator.eval()
        print(f"Running evaluation plots for epoch {epoch}...")

        plot_cfg = self.config['logging']
        plot_dir = os.path.join(self.config['data']['output_dir'], 'plots')
        # we no longer crop from the dataset itself
        #   original_crop_len = self.eval_dataset.dataset.crop_len
        #  self.eval_dataset.dataset.crop_len = None

        with torch.no_grad():
            num_eval_samples = min(plot_cfg['num_plot_examples'], len(self.eval_dataset))
            eval_indices = random.sample(range(len(self.eval_dataset)), num_eval_samples)

            log_dict = {}
            for i, idx in enumerate(eval_indices):
                actual_idx = self.eval_dataset.indices[idx]
                sample, mel_len, filename = self.eval_dataset.dataset[actual_idx]

                sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(self.device)
                mel_len_tensor = torch.tensor([mel_len], dtype=torch.int32).to(self.device)

                with autocast(enabled=self.device.type == 'cuda',
                              dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
                    recon_pre, recon_post = self.generator(sample_tensor, mel_len_tensor)

                sample_to_plot = sample_tensor[0, :mel_len, :]
                recon_pre_to_plot = recon_pre[0, :mel_len, :]
                recon_post_to_plot = recon_post[0, :mel_len, :]

                vmin = min(sample_to_plot.min().item(), recon_pre_to_plot.min().item(), recon_post_to_plot.min().item())
                vmax = max(sample_to_plot.max().item(), recon_pre_to_plot.max().item(), recon_post_to_plot.max().item())

                save_path_orig = os.path.join(plot_dir,
                                              f'epoch_{epoch:03d}_eval_orig_{i + 1}_{os.path.splitext(filename)[0]}.png')
                save_path_recon_pre = os.path.join(plot_dir,
                                                   f'epoch_{epoch:03d}_eval_recon_pre_{i + 1}_{os.path.splitext(filename)[0]}.png')
                save_path_recon_post = os.path.join(plot_dir,
                                                    f'epoch_{epoch:03d}_eval_recon_post_{i + 1}_{os.path.splitext(filename)[0]}.png')

                log_dict[f"eval_comparison_{i + 1}"] = plot_mel_spectrograms(
                    [sample_to_plot, recon_pre_to_plot, recon_post_to_plot],
                    ['Original', 'Reconstructed (Pre-Refiner)', 'Reconstructed (Post-Refiner)'],
                    vmin, vmax, save_path_orig, f'Epoch {epoch} Eval - {os.path.splitext(filename)[0]}'
                )
            wandb.log(log_dict)

        # self.eval_dataset.dataset.crop_len = original_crop_len
        self.generator.train()

    def _save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.config['data']['output_dir'], f'checkpoint_epoch_{epoch:03d}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scaler_g_state_dict': self.scaler_g.state_dict(),
            'scaler_d_state_dict': self.scaler_d.state_dict(),
            'config': self.config
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    def train(self):
        if self.train_loader is None:
            print("Training dataloader is not available. Skipping training.")
            return

        for epoch in range(self.start_epoch, self.config['training']['num_epochs'] + 1):
            self._train_epoch(epoch)

            if epoch % self.config['logging']['eval_interval'] == 0 and self.eval_dataset is not None:
                self._evaluate(epoch)

            if epoch % self.config['logging']['save_interval'] == 0:
                self._save_checkpoint(epoch)

        print("Training finished.")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Train an MQGAN model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to a pretrained checkpoint to load.')
    parser.add_argument('--output_dir', type=str, default=None, help="Path to the output directory, overriding config.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.pretrained:
        config['training']['pretrained'] = args.pretrained
    
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir

    trainer = Trainer(config)
    trainer.train()
    wandb.finish()


if __name__ == '__main__':
    main()
