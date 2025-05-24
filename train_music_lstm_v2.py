"""
train_lstm_music_enhanced.py
----------------------------
Enhanced training script for genre-conditioned next-token prediction on music-token
sequences. Model: 2-layer LSTM (1024 hidden) + projection, with a learned genre
embedding added to token embeddings.

Adds:
- Validation loop for evaluation loss.
- Weights & Biases (WandB) integration for logging.
- Mixed Precision training with bfloat16 (bf16) via torch.cuda.amp.

Data format comes from:
    ① *.npy files – 1-D int arrays of token IDs (BOS already prepended)
    ② fname_to_id.json – maps each chunk file to its GENREID
"""

import argparse
import json
import math
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision
from tqdm import tqdm
import wandb # For logging

# Check CUDA availability for mixed precision
IS_CUDA_AVAILABLE = torch.cuda.is_available()
if IS_CUDA_AVAILABLE:
    # Set benchmark to False for reproducibility if needed, True might speed up
    # training if input sizes don't vary much. Let's keep it False as per original.
    torch.backends.cudnn.benchmark = False
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    # Check bf16 support
    try:
        # Attempt a small operation in bf16
        _ = torch.randn(1, device='cuda').to(torch.bfloat16) * torch.randn(1, device='cuda').to(torch.bfloat16)
        IS_BF16_SUPPORTED = True
        print("BF16 is supported on this device.")
    except RuntimeError:
        IS_BF16_SUPPORTED = False
        print("BF16 is NOT supported on this device. Mixed precision will use FP32.")

else:
    IS_BF16_SUPPORTED = False
    print("CUDA not available. Running on CPU. Mixed precision disabled.")


# ---------- Dataset & Collate (same as before) ----------
class MusicChunkDataset(Dataset):
    """
    Loads *.npy chunks that contain 1-D int arrays of token indices and their genre ID.
    A <BOS> token is prepended; sequences are padded inside the collate_fn.
    """
    def __init__(
        self,
        chunks_dir: str | Path,
        mapping_json: str | Path,
        bos_id: int = 1,
        pad_id: int = 0,
    ):
        self.chunks_dir = Path(chunks_dir)
        self.bos_id = bos_id
        self.pad_id = pad_id

        # map filename → genreID
        print(f"Loading mapping from: {mapping_json}")
        with open(mapping_json, "r", encoding="utf-8") as f:
            fname2genre = json.load(f)
        print(f"Found {len(fname2genre)} entries in mapping file.")

        # keep only files that actually exist in chunks_dir
        print(f"Scanning for files in: {self.chunks_dir}")
        self.items: List[Tuple[Path, int]] = [
            (self.chunks_dir / fname, gid)
            for fname, gid in tqdm(fname2genre.items(), desc="Checking files")
            if (self.chunks_dir / fname).is_file()
        ]

        if not self.items:
            raise RuntimeError(f"No matching .npy files found in {self.chunks_dir} based on {mapping_json}!")
        print(f"Found {len(self.items)} valid chunk files.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        fpath, genre_id = self.items[idx]
        try:
            seq = np.load(fpath).astype(np.int64)  # (T,)
            seq = np.insert(seq, 0, self.bos_id)  # prepend BOS
            seq = torch.from_numpy(seq)
            # filename pattern: XXXXX_chunkYYY.npy  → extract YYY (as int)
            chunk_id = int(fpath.stem.split('_')[-1].replace('chunk', ''))
            
            return seq, genre_id, chunk_id
        except Exception as e:
            print(f"Error loading or processing file {fpath}: {e}")
            # Handle error: return a dummy item or raise an exception
            # Returning a dummy item might skew training if not handled properly.
            # For simplicity here, we re-raise, but a robust pipeline might skip.
            raise e


def collate_music(batch: List[Tuple[torch.Tensor, int, int]], pad_id: int = 0):
    """
    Pads variable-length sequences to the max length in the batch.
    Returns: tokens (B, L), genre_ids (B,), lengths (B,)
    """
    # Filter out potential errors if __getitem__ handled them gracefully (e.g., returned None)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None # Indicate an empty batch

    seqs, genres, chunk_ids_raw = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)

    # Handle edge case where all sequences might be empty after BOS prepending
    if lengths.numel() == 0 or lengths.max().item() == 0:
         # Create minimal tensors to avoid errors downstream
         max_len = 1 # At least BOS
    else:
         max_len = lengths.max().item()

    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)

    for i, seq in enumerate(seqs):
        # Ensure seq is not empty before padding
        if len(seq) > 0:
             padded[i, : len(seq)] = seq

    genre_ids = torch.tensor(genres, dtype=torch.long)
    chunk_ids  = torch.tensor(chunk_ids_raw, dtype=torch.long)
    return padded, genre_ids, lengths, chunk_ids


# ---------- Model (same as before) ---------------------------------
class MusicLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_genres: int,
        emb_dim: int = 512,
        lstm_hid: int = 1024,
        lstm_layers: int = 2,
        pad_id: int = 0,
        drop=0.1,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.tok_emb   = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.genre_emb = nn.Embedding(num_genres, emb_dim)
        self.dropout = nn.Dropout(drop)

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hid,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(lstm_hid, vocab_size)
        print(f"Model initialized: Vocab={vocab_size}, Genres={num_genres}, Emb={emb_dim}, LSTM_Hid={lstm_hid}, Layers={lstm_layers}")
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {num_params / 1_000_000:.2f} M")


    def forward(self, tokens, genre_ids, lengths):
        """
        tokens   : (B, L)
        genre_ids: (B,)
        lengths  : (B,)  valid lengths including BOS
        """
        tok_e = self.tok_emb(tokens)                           # (B, L, D)
        gen_e = self.genre_emb(genre_ids)[:, None, :]          # (B, 1, D)
        x = tok_e + gen_e                             # broadcast add

        # Ensure lengths are valid (at least 1) before packing
        valid_lengths = torch.clamp(lengths, min=1)

        # Move lengths to CPU for pack_padded_sequence as required
        packed = nn.utils.rnn.pack_padded_sequence(
            x, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        # Pad sequence back
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, padding_value=0.0 # Use 0.0 for padding features
        )                                                    # (B, L, H)

        out = self.dropout(out)
        logits = self.proj(out)                              # (B, L, V)
        
        return logits


# ---------- Loss (using built-in CrossEntropyLoss is fine) ----------
# The original train loop already used nn.CrossEntropyLoss(ignore_index=pad_id),
# which is efficient and equivalent to the masked_ce_loss function. We'll stick with that.

# ---------- Training Loop (Modified) -------------------------------
def train_loop(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler], # Optional GradScaler for AMP
    use_amp: bool,                # Flag to enable AMP context
    epoch: int,
    pad_id: int,
    log_every: int = 100,
    wandb_log: bool = True,
    global_step: int = 0,
    max_grad_norm: float = 1.0
) -> Tuple[float, int]:
    """Runs one epoch of training."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    last_log_time = time.time()

    # Determine AMP dtype
    amp_dtype = torch.bfloat16 if use_amp and IS_BF16_SUPPORTED else torch.float16
    if use_amp and not IS_BF16_SUPPORTED and device.type == 'cuda':
        print("Warning: BF16 not supported, using FP16 for mixed precision.")
    elif use_amp and device.type == 'cpu':
        print("Warning: Mixed precision requested on CPU, disabling AMP.")
        use_amp = False # Disable AMP on CPU

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for step, batch_data in enumerate(pbar, 1):
        # Handle potential empty batches from collate_fn
        if batch_data[0] is None:
            print(f"Skipping empty batch at step {step}")
            continue

        tok, genres, lengths, chunks = batch_data
        tok, genres, chunks = tok.to(device), genres.to(device), chunks.to(device)
        # lengths stay on CPU for pack_padded_sequence

        # Input/Target shift for next-token prediction
        # Ensure lengths are at least 1 for slicing
        valid_mask = lengths >= 1
        if not valid_mask.any(): # Skip batch if all sequences are empty
             print(f"Skipping batch {step} due to all sequences being length 0.")
             continue

        # Filter batch based on valid lengths before processing
        tok = tok[valid_mask]
        genres = genres[valid_mask]
        lengths = lengths[valid_mask]

        # Adjust lengths and targets for next-token prediction (predict token t+1 from token t)
        # We need lengths >= 2 to have a valid target after BOS
        pred_mask = lengths >= 2
        if not pred_mask.any():
             print(f"Skipping batch {step} as no sequences have length >= 2 for prediction.")
             continue

        # Filter again for prediction pairs
        tok = tok[pred_mask]
        genres = genres[pred_mask]
        lengths = lengths[pred_mask]

        # inp: tokens up to T-1, tgt: tokens from 1 to T
        inp, tgt = tok[:, :-1], tok[:, 1:]
        # lengths for LSTM should be the length of the input sequence `inp`
        input_lengths = lengths - 1

        # Skip if inputs become empty after slicing
        if inp.shape[1] == 0:
            print(f"Skipping batch {step} because input sequence length became 0 after slicing.")
            continue

        optimizer.zero_grad(set_to_none=True)

        # Mixed Precision Context
        with autocast(enabled=use_amp, dtype=amp_dtype if device.type == 'cuda' else torch.float32):
            logits = model(inp, genres, input_lengths)  # (B, L-1, V)
            B, L_pred, V = logits.shape

            # Reshape for CrossEntropyLoss: (B * L_pred, V) and (B * L_pred,)
            # Target tensor `tgt` already has shape (B, L-1)
            loss = criterion(logits.reshape(B * L_pred, V), tgt.reshape(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected at step {global_step + step}. Skipping backward pass for this batch.")
            # Optionally: log details, save batch for debugging
            # wandb.log({"error": "NaN/Inf loss"}, step=global_step + step)
            continue # Skip optimizer step for this batch

        # Scale loss and backpropagate if using AMP scaler
        if scaler is not None:
            scaler.scale(loss).backward()
            # Unscale gradients before clipping (recommended)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else: # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss
        num_batches += 1

        # Logging
        pbar.set_postfix(loss=f"{current_loss:.4f}")
        if step % log_every == 0:
            avg_loss_interval = total_loss / num_batches # Or calculate over the interval if preferred
            current_time = time.time()
            elapsed = current_time - last_log_time
            steps_per_sec = log_every / elapsed if elapsed > 0 else 0
            print(f"\nEpoch {epoch} | Step {step:5d} | Avg Train Loss (epoch): {avg_loss_interval:.4f} | Steps/sec: {steps_per_sec:.2f}")
            if wandb_log:
                wandb.log({
                    "train/loss_step": current_loss,
                    "train/steps_per_sec": steps_per_sec,
                    "step": global_step + step,
                    "epoch": epoch + (step / len(dataloader)) # Fractional epoch
                })
            last_log_time = time.time()

    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_epoch_loss, global_step + num_batches # Return avg loss and updated global step


# ---------- Evaluation Loop (New) ----------------------------------
@torch.no_grad() # Disable gradient calculations
def evaluate_loop(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,           # Flag to enable AMP context
    epoch: int,
    pad_id: int
) -> float:
    """Runs evaluation on the validation set."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    # Determine AMP dtype
    amp_dtype = torch.bfloat16 if use_amp and IS_BF16_SUPPORTED else torch.float16
    if use_amp and device.type == 'cpu':
        use_amp = False # Disable AMP on CPU

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Evaluating")
    for batch_data in pbar:
        # Handle potential empty batches
        if batch_data[0] is None:
            continue

        tok, genres, lengths = batch_data
        tok, genres = tok.to(device), genres.to(device)
        # lengths stay on CPU

        # Input/Target shift (same logic as training)
        valid_mask = lengths >= 1
        if not valid_mask.any(): continue
        tok, genres, lengths = tok[valid_mask], genres[valid_mask], lengths[valid_mask]

        pred_mask = lengths >= 2
        if not pred_mask.any(): continue
        tok, genres, lengths = tok[pred_mask], genres[pred_mask], lengths[pred_mask]

        inp, tgt = tok[:, :-1], tok[:, 1:]
        input_lengths = lengths - 1

        if inp.shape[1] == 0: continue

        # Mixed Precision Context (only for forward pass in eval)
        with autocast(enabled=use_amp, dtype=amp_dtype if device.type == 'cuda' else torch.float32):
            logits = model(inp, genres, input_lengths)
            B, L_pred, V = logits.shape
            loss = criterion(logits.reshape(B * L_pred, V), tgt.reshape(-1))

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1
        else:
             print(f"Warning: NaN or Inf loss detected during evaluation. Skipping batch.")


    avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_val_loss


# ---------- CLI / main (Modified) ----------------------------------
def main():
    p = argparse.ArgumentParser(description="Train a genre-conditioned LSTM music model.")
    # Data args
    p.add_argument("--chunks_dir", default="musicmels", help="Directory containing *.npy chunk files.")
    p.add_argument("--mapping_json", default="fname_to_id.json", help="JSON mapping filenames to genre IDs.")
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation (e.g., 0.1 for 10%).")
    # Model args
    p.add_argument("--vocab_size", type=int, required=True, help="Size of the token vocabulary.")
    p.add_argument("--num_genres", type=int, required=True, help="Number of unique genres.")
    p.add_argument("--emb_dim", type=int, default=512, help="Dimension of token and genre embeddings.")
    p.add_argument("--lstm_hid", type=int, default=1024, help="Hidden dimension size of LSTM layers.")
    p.add_argument("--lstm_layers", type=int, default=2, help="Number of LSTM layers.")
    p.add_argument("--bos_id", type=int, default=1, help="Token ID for Beginning-Of-Sequence.")
    p.add_argument("--pad_id", type=int, default=0, help="Token ID for Padding.")
    # Training args
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size per device.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW optimizer.")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm value.")
    p.add_argument("--device", default="cuda" if IS_CUDA_AVAILABLE else "cpu", help="Device to use ('cuda' or 'cpu').")
    p.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes.")
    p.add_argument("--log_every", type=int, default=100, help="Log training loss every N steps.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    # Mixed Precision args
    p.add_argument("--mixed_precision", action="store_true", help="Enable BFloat16 mixed precision training (requires CUDA with BF16 support). Falls back to FP16 if BF16 not available.")
    # WandB args
    p.add_argument("--wandb_project", type=str, default="music-lstm", help="WandB project name.")
    p.add_argument("--wandb_entity", type=str, default=None, help="WandB entity (username or team).")
    p.add_argument("--wandb_name", type=str, default=None, help="WandB run name (defaults to auto-generated).")
    p.add_argument("--no_wandb", action="store_true", help="Disable WandB logging.")
    p.add_argument("--out_dir", type=str, default="logs/musiclstm-run1", help="Run out dir")

    args = p.parse_args()
    

    # Seed setting for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if IS_CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Setup device
    device = torch.device(args.device)
    if args.device == "cuda" and not IS_CUDA_AVAILABLE:
        print("Warning: CUDA requested but not available. Switching to CPU.")
        device = torch.device("cpu")
        args.device = "cpu" # Update args to reflect actual device

    # Determine if AMP should be used
    use_amp = args.mixed_precision and device.type == 'cuda'
    if use_amp and not (IS_BF16_SUPPORTED or torch.cuda.is_bf16_supported()): # Double check bf16 support
         print("Warning: Requested BF16 mixed precision, but it's not supported. Trying FP16.")
         if not torch.cuda.is_available() or not hasattr(torch.cuda, 'amp') or not torch.cuda.get_device_capability()[0] >= 7:
              print("Warning: FP16 mixed precision also not well supported. Disabling mixed precision.")
              use_amp = False
         else:
              # Use FP16 if BF16 isn't available but FP16 likely is
              print("Using FP16 mixed precision.")
    elif args.mixed_precision and device.type == 'cpu':
         print("Mixed precision is only supported on CUDA. Disabling.")
         use_amp = False

    print(f"Using device: {device}")
    print(f"Mixed Precision (AMP) enabled: {use_amp}")

    # Initialize WandB
    if not args.no_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                config=vars(args) # Log all hyperparameters
            )
            wandb_log = True
            print("WandB initialized.")
        except Exception as e:
            print(f"Could not initialize WandB: {e}. Disabling WandB logging.")
            wandb_log = False
    else:
        wandb_log = False
        print("WandB logging disabled.")

    # --- Data Loading and Splitting ---
    print("Loading dataset...")
    full_dataset = MusicChunkDataset(
        args.chunks_dir, args.mapping_json, bos_id=args.bos_id, pad_id=args.pad_id
    )

    # Split dataset
    total_size = len(full_dataset)
    val_size = int(args.val_split * total_size)
    train_size = total_size - val_size

    if val_size == 0 or train_size == 0:
         raise ValueError(f"Validation split {args.val_split} resulted in zero samples for train or val. Adjust split or check dataset size ({total_size}).")

    print(f"Splitting dataset: Train={train_size}, Validation={val_size}")
    # Use generator for reproducibility with seed
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Create DataLoaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False, # Pin memory only for CUDA
        collate_fn=lambda b: collate_music(b, pad_id=args.pad_id),
        persistent_workers=True if args.num_workers > 0 else False # Helps speed up epoch starts
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2, # Often possible to use larger batch for eval
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=lambda b: collate_music(b, pad_id=args.pad_id),
        persistent_workers=True if args.num_workers > 0 else False
    )
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # --- Model, Optimizer, Loss ---
    model = MusicLSTM(
        vocab_size=args.vocab_size,
        num_genres=args.num_genres,
        emb_dim=args.emb_dim,
        lstm_hid=args.lstm_hid,
        lstm_layers=args.lstm_layers,
        pad_id=args.pad_id,
    ).to(device)

    if wandb_log:
        wandb.watch(model, log="all", log_freq=args.log_every * 10) # Watch model gradients/parameters

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_id)

    # Initialize GradScaler for Mixed Precision (only if use_amp is True)
    scaler = GradScaler(enabled=use_amp) if device.type == 'cuda' else None
    if use_amp and scaler:
         print("Initialized GradScaler for AMP.")

    # --- Training & Evaluation Loop ---
    print(f"\nStarting training for {args.epochs} epochs...")
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        # Training
        avg_train_loss, global_step = train_loop(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch=epoch,
            pad_id=args.pad_id,
            log_every=args.log_every,
            wandb_log=wandb_log,
            global_step=global_step,
            max_grad_norm=args.max_grad_norm
        )

        # Evaluation
        avg_val_loss = evaluate_loop(
            model=model,
            dataloader=val_dl,
            criterion=criterion,
            device=device,
            use_amp=use_amp, # Use autocast in eval but no grad scaling
            epoch=epoch,
            pad_id=args.pad_id
        )

        # Perplexity (lower is better)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 700 else float('inf') # Avoid overflow
        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 700 else float('inf')

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f} | Train Perplexity: {train_ppl:.2f}")
        print(f"  Avg Val Loss  : {avg_val_loss:.4f} | Val Perplexity  : {val_ppl:.2f}")

        # Log epoch metrics to WandB
        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "train/loss_epoch": avg_train_loss,
                "train/perplexity": train_ppl,
                "val/loss": avg_val_loss,
                "val/perplexity": val_ppl,
                "global_step": global_step # Log final step count for the epoch
            })

        # Simple best model saving based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            file_name = f"model_epoch_{epoch}_valloss_{avg_val_loss:.4f}.pt"
            out_path = os.path.join(args.out_dir, file_name)
            save_path = Path(out_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'args': args # Save args for reproducibility
            }, save_path)
            print(f"  New best validation loss. Saved model to {save_path}")
            if wandb_log:
                 wandb.save(str(save_path)) # Save best model checkpoints to WandB

    print("\nTraining finished.")
    if wandb_log:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()