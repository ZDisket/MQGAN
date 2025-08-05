import torch
torch.backends.cudnn.benchmark = False

import argparse
import os
import yaml
from preencoder import get_pre_encoder, sequence_mask
from scripted_preencoder import ScriptedPreEncoder

def main():
    """
    Main function to handle model conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a PreEncoder checkpoint to a single TorchScript model with traced encode/decode methods.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the input PyTorch checkpoint file (.pth).')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the model configuration YAML file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output folder where the models and config will be saved.')
    args = parser.parse_args()

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # --- Load Config ---
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        f.seek(0)
        config_str = f.read()

    model_params = config['model']
    gen_params = model_params['generator']
    mel_channels = model_params['mel_channels']

    # --- Trace and Save Function ---
    def trace_and_save_model(device):
        print(f"\n--- Processing for device: {device} ---")
        model = get_pre_encoder(
            model_path=args.checkpoint,
            device=device,
            mel_channels=mel_channels,
            channels=gen_params['channels'],
            kernel_sizes=gen_params['kernel_sizes'],
            fsq_levels=gen_params['fsq_levels']
        )

        # --- Dummy Inputs for Tracing ---
        trace_len = 512
        print(f"Using fixed sequence length for tracing: {trace_len}")
        dummy_spec = torch.randn(1, trace_len, mel_channels, device=device)
        dummy_lengths = torch.tensor([trace_len], device=device, dtype=torch.long)
        dummy_mask = sequence_mask(trace_len, dummy_lengths).unsqueeze(1)
        dummy_indices = torch.randint(0, model.codebook_size, (1, trace_len), device=device, dtype=torch.long)

        # --- Trace Module ---
        try:
            print("Tracing the PreEncoder module for 'encode' and 'decode' methods...")
            traced_model = torch.jit.trace_module(
                model,
                {
                    'encode': (dummy_spec, dummy_mask),
                    'decode': (dummy_indices, dummy_mask)
                }
            )
            model_path = os.path.join(args.output_dir, f'model_{device}.pt')
            print(f"Saving {device.upper()} TorchScript model to: {model_path}")
            traced_model.save(model_path)
            print("Model saved.")
        except Exception as e:
            print(f"An error occurred during torch.jit.trace_module on {device}: {e}")

    # --- CPU Model Export ---
    trace_and_save_model('cpu')

    # --- CUDA Model Export ---
    if torch.cuda.is_available():
        trace_and_save_model('cuda')
    else:
        print("\nSkipping CUDA export: CUDA is not available.")

    # --- Save Config File ---
    config_path = os.path.join(args.output_dir, 'model_config.yaml')
    print(f"\nSaving configuration file to: {config_path}")
    with open(config_path, 'w') as f:
        f.write(config_str)
    print("Configuration file saved.")

    # --- Verification Step (CUDA Model) ---
    if torch.cuda.is_available() and os.path.exists(os.path.join(args.output_dir, 'model_cuda.pt')):
        print("\n--- Verifying the exported CUDA TorchScript model ---")
        try:
            wrapper = ScriptedPreEncoder(args.output_dir, device='cuda')

            # Create a dummy batch with a fixed length that doesnt match tracing
            eval_len = 384
            spec = torch.randn(2, eval_len, wrapper.mel_channels, device='cuda')
            lengths = torch.tensor([eval_len, eval_len], device='cuda', dtype=torch.long)

            print(f"Verifying with input shape: {spec.shape}")

            # Test encode and decode pipeline
            print("Testing encode -> decode pipeline...")
            indices = wrapper.encode(spec, lengths=lengths)
            reconstructed_batch = wrapper.decode(indices, lengths=lengths)

            assert reconstructed_batch.shape == spec.shape, f"Shape mismatch after decode. Expected {spec.shape}, got {reconstructed_batch.shape}"
            print("Verification successful! The traced model handles the encode/decode pipeline correctly.")

        except Exception as e:
            print(f"\nAn error occurred during the verification step: {e}")

    print("\nConversion process finished.")

if __name__ == '__main__':
    main()
