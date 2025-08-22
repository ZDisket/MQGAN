import torch
import numpy as np
import argparse
import os
import yaml
from tqdm import tqdm
from preencoder import get_pre_encoder, sequence_mask

def reencode_spectrograms(checkpoint_path, config, input_dir, output_dir, device, batch_size):
    """
    Loads a raw PreEncoder model from a checkpoint and uses it to re-encode all .npy spectrograms
    found in a given directory using batch processing.

    Args:
        checkpoint_path (str): Path to the PyTorch model checkpoint file (.pth).
        config (dict): The model configuration dictionary.
        input_dir (str): Path to the root directory containing the original .npy spectrograms.
        output_dir (str): Path to the directory where the re-encoded .npy files will be saved.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        batch_size (int): The number of files to process in a single batch.
    """
    # 1. Load the PreEncoder model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        model_params = config['model']
        gen_params = model_params['generator']
        model = get_pre_encoder(
            model_path=checkpoint_path,
            device=device,
            mel_channels=model_params['mel_channels'],
            channels=gen_params['channels'],
            kernel_sizes=gen_params['kernel_sizes'],
            fsq_levels=gen_params['fsq_levels'],
            refiner_base_channels=gen_params.get('refiner_base_channels', 128),
            refiner_depth=gen_params.get('refiner_depth', 3),
            refiner_hidden_proj_divisor=gen_params.get('refiner_hidden_proj_divisor', 8),
            inference=True,
        )
    except (FileNotFoundError, RuntimeError, KeyError) as e:
        print(f"Error: Could not load the model. {e}")
        return

    # 2. Find all .npy files recursively
    print(f"Searching for .npy files in: {input_dir}")
    npy_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))

    if not npy_files:
        print("Warning: No .npy files were found.")
        return

    print(f"Found {len(npy_files)} spectrogram files to process.")

    # 3. Create batches of file paths
    file_batches = [npy_files[i:i + batch_size] for i in range(0, len(npy_files), batch_size)]

    # 4. Process each batch
    for batch_paths in tqdm(file_batches, desc="Re-encoding Spectrograms"):
        try:
            # Load spectrograms and get original lengths
            spectrograms = [np.load(p) for p in batch_paths]
            original_lengths = [s.shape[0] for s in spectrograms]
            max_len = max(original_lengths)

            # Pad each spectrogram to the max length in the batch
            padded_spectrograms = []
            for spec in spectrograms:
                pad_len = max_len - spec.shape[0]
                padding = np.zeros((pad_len, spec.shape[1]), dtype=spec.dtype)
                padded_spec = np.concatenate([spec, padding], axis=0)
                padded_spectrograms.append(padded_spec)

            # Stack into a single tensor for batch processing
            batch_tensor = torch.from_numpy(np.stack(padded_spectrograms)).float().to(device)
            lengths_tensor = torch.tensor(original_lengths, dtype=torch.long, device=device)
            
            # Create the mask for padded sequences
            mask = sequence_mask(max_len, lengths_tensor).unsqueeze(1)

            # Encode and decode the entire batch, providing the original lengths
            with torch.no_grad():
                indices = model.encode(batch_tensor, x_mask=mask)
                reencoded_batch = model.decode(indices, x_mask=mask)

            # Save each re-encoded spectrogram from the batch
            for i, reencoded_tensor in enumerate(reencoded_batch):
                original_len = original_lengths[i]
                file_path = batch_paths[i]

                # Trim the padding to restore the original length
                trimmed_tensor = reencoded_tensor[:original_len, :]
                reencoded_np = trimmed_tensor.cpu().numpy()

                # Create the mirrored output path and save the file
                relative_path = os.path.relpath(file_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, reencoded_np)

        except Exception as e:
            print(f"\nCould not process batch starting with {batch_paths[0]}. Error: {e}")
            continue

    print("\nProcessing complete.")
    print(f"Re-encoded spectrograms have been saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Re-encode spectrograms using a raw PreEncoder model checkpoint. "
                    "This is useful for generating training data for a vocoder.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the raw PyTorch PreEncoder model checkpoint (.pth).')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input folder containing .npy spectrograms.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output folder where re-encoded spectrograms will be saved.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for inference (e.g., "cpu", "cuda"). Defaults to \'cpu\'.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of spectrograms to process in a single batch. Defaults to 32.')

    args = parser.parse_args()

    # Load config from YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config file: {e}")
        return

    reencode_spectrograms(args.checkpoint, config, args.input_dir, args.output_dir, args.device, args.batch_size)

if __name__ == '__main__':
    main()
