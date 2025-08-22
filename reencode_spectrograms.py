import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from scripted_preencoder import ScriptedPreEncoder

def reencode_spectrograms(model_path, input_dir, output_dir, device, batch_size):
    """
    Loads a TorchScript PreEncoder model and uses it to re-encode all .npy spectrograms
    found in a given directory using batch processing for improved performance.

    Args:
        model_path (str): Path to the TorchScript-exported model file (.pt).
        input_dir (str): Path to the root directory containing the original .npy spectrograms.
        output_dir (str): Path to the directory where the re-encoded .npy files will be saved.
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        batch_size (int): The number of files to process in a single batch.
    """
    # 1. Load the ScriptedPreEncoder model
    print(f"Loading model from: {model_path}")
    try:
        model = ScriptedPreEncoder(model_path, device=device)
    except (FileNotFoundError, RuntimeError) as e:
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
            batch_tensor = torch.from_numpy(np.stack(padded_spectrograms)).float()

            # Encode and decode the entire batch, providing the original lengths
            indices = model.encode(batch_tensor, lengths=original_lengths)
            reencoded_batch = model.decode(indices, lengths=original_lengths)

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
        description="Re-encode spectrograms using a TorchScript PreEncoder model. "
                    "This is useful for generating training data for a vocoder.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the TorchScript-exported PreEncoder model folder')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the input folder containing .npy spectrograms.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output folder where re-encoded spectrograms will be saved.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for inference (e.g., "cpu", "cuda"). Defaults to \'cpu\'.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of spectrograms to process in a single batch. Defaults to 32.')

    args = parser.parse_args()

    reencode_spectrograms(args.model, args.input_dir, args.output_dir, args.device, args.batch_size)

if __name__ == '__main__':
    main()
