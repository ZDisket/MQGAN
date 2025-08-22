# MQGAN: Mel Quantization Generative Adversarial Network

This repository contains the implementation of (MQGAN) for audio synthesis. The project is structured to facilitate the entire workflow from data preparation to model deployment.

## Table of Contents

1.  [Setup](#1-setup)
2.  [Data Preprocessing](#2-data-preprocessing)
    *   [Converting Audio to Mel Spectrograms](#converting-audio-to-mel-spectrograms)
    *   [Re-encoding Spectrograms (Optional)](#re-encoding-spectrograms-optional)
3.  [Training the PreEncoder Model](#3-training-the-preencoder-model)
4.  [Exporting to TorchScript](#4-exporting-to-torchscript)
5.  [Pretrained Models](#5-pretrained-models)

---

## 1. Setup

Before you begin, ensure you have Python 3.9+ installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/MQGAN.git
    cd MQGAN
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Data Preprocessing

The MQGAN model operates on Mel spectrograms. You'll need to convert your raw audio files into this format.

### Converting Audio to Mel Spectrograms

Use `convert_spectrograms.py` to transform your audio files (e.g., WAV, FLAC) into Mel spectrograms, saved as `.npy` files.

1.  **Prepare your audio data:** Place your audio files in an input directory. The script will mirror the directory structure in the output.

2.  **Configure spectrogram extraction:**
    Edit the `spec_config.yaml` file to define parameters for Mel spectrogram extraction, such as `sampling_rate`, `n_mel_channels`, `filter_length`, `hop_length`, etc. You can find some at configs/

3.  **Run the conversion script:**
    ```bash
    python convert_spectrograms.py --config configs/spec_config_hifispeech.yaml
    ```
    You can override `input_folder` and `output_folder` directly from the command line:
    ```bash
    python convert_spectrograms.py --config configs/spec_config_hifispeech.yaml --input_folder /data/raw_audio --output_folder /data/mels
    ```
    The script will create `.npy` files in the specified `output_folder`, preserving the original directory structure.

### Re-encoding Spectrograms (Optional)

After training the `PreEncoder` model, you might want to re-encode your original Mel spectrograms using the trained model. This is useful for generating training data for a subsequent vocoder or for analyzing the quantized representation.

You have two options for re-encoding:

*   **Using a TorchScript-exported model:** If you have already exported your `PreEncoder` to TorchScript (see [Exporting to TorchScript](#4-exporting-to-torchscript)), use `reencode_spectrograms.py`.
    ```bash
    python reencode_spectrograms.py \
        --model /path/to/your/exported_model_folder \
        --input_dir /path/to/your/original_mels \
        --output_dir /path/to/save/reencoded_mels \
        --device cuda # or cpu
    ```
    The `--model` argument should point to the directory containing `model_cpu.pt` (and optionally `model_cuda.pt`) and `model_config.yaml`.

*   **Using a raw PyTorch checkpoint:** If you prefer to use a raw `.pth` checkpoint and its corresponding `config.yaml` directly, use `reencode_spectrograms_from_checkpoint.py`.
    ```bash
    python reencode_spectrograms_from_checkpoint.py \
        --checkpoint /path/to/your/model.pth \
        --config /path/to/your/model_config.yaml \
        --input_dir /path/to/your/original_mels \
        --output_dir /path/to/save/reencoded_mels \
        --device cuda # or cpu
    ```
    The `--config` argument here refers to the model's configuration file (e.g., `model_config_hifimusic.yaml` or `model_config_hifispeech.yaml`), not `spec_config.yaml`.

## 3. Training the PreEncoder Model

The `PreEncoder` model is trained using `train.py`. This script handles the GAN training loop, including the generator (PreEncoder) and discriminators.

1.  **Prepare your training data:** Ensure you have generated Mel spectrograms using `convert_spectrograms.py` and they are located in the `data_dir` specified in your training configuration.

2.  **Configure training parameters:**
    Edit a training configuration file (e.g., `model_config_hifimusic.yaml` or `model_config_hifispeech.yaml`). This file defines model architecture, training hyperparameters, loss weights, and data paths.

3.  **Start training:**
    ```bash
    python train.py --config configs/model_config_hifimusic.yaml
    ```
    You can resume training from a checkpoint:
    ```bash
    python train.py --config configs/model_config_hifimusic.yaml --pretrained checkpoints/music_preencoder/checkpoint_epoch_050.pth
    ```
    Training progress, losses, and example spectrograms will be logged to Weights & Biases (WandB).

## 4. Exporting to TorchScript

Once your `PreEncoder` model is trained, you can export it to TorchScript for easier deployment and faster inference.

1.  **Run the conversion script:**
    ```bash
    python convert_to_torchscript.py \
        --checkpoint /path/to/your/trained_model.pth \
        --config /path/to/your/model_config.yaml \
        --output_dir /path/to/save/exported_model
    ```
    *   `--checkpoint`: Path to the `.pth` checkpoint file from your training run.
    *   `--config`: Path to the model's configuration YAML file (e.g., `model_config_hifimusic.yaml`) that was used for training.
    *   `--output_dir`: Directory where the TorchScript models (`model_cpu.pt`, `model_cuda.pt`) and a copy of the config (`model_config.yaml`) will be saved.

2.  **Using the exported model:**
    The `scripted_preencoder.py` file provides a `ScriptedPreEncoder` class to easily load and use the exported TorchScript model:
    ```python
    from scripted_preencoder import ScriptedPreEncoder
    import torch
    import numpy as np

    # Load the exported model
    model_wrapper = ScriptedPreEncoder("/path/to/save/exported_model", device='cuda') # or 'cpu'

    # Example usage:
    # Assuming 'mel_input_np' is a NumPy array of your Mel spectrogram (batch, seq_len, mel_channels)
    mel_input_tensor = torch.from_numpy(mel_input_np).float()
    lengths = torch.tensor([mel_input_np.shape[1]]) # Example for a single spectrogram

    # Encode to discrete tokens
    indices = model_wrapper.encode(mel_input_tensor, lengths=lengths)
    print(f"Encoded indices shape: {indices.shape}")

    # Decode back to spectrogram
    reconstructed_mel = model_wrapper.decode(indices, lengths=lengths)
    print(f"Reconstructed mel shape: {reconstructed_mel.shape}")
    ```

## 5. Pretrained Models

We provide a selection of pretrained MQGAN models for different audio domains. These models include both the PreEncoder (quantizer) and the iSTFTNet components.

| Model Name             | Sampling Rate | Mel params (channels, fmin-max) | Link to Pretrained Models (Quantizer & iSTFTNet)                            | Colab Notebook Example |
|:-----------------------| :--- |:--------------------------------|:----------------------------------------------------------------------------| :--- |
| *MQGAN+R-HifiSpeech-1* | *44.1 kHz* | *120, 0-22050 Hz*               | *[MQGAN](https://huggingface.co/ZDisket/mqganplusR-hifispeech), [ISTFTNet](https://huggingface.co/ZDisket/istftnet-hifispeech)* | *[Colab Link](https://colab.research.google.com/drive/1aXQtm1ZoIe8ACMV_Vgb-M9PZVQjuvhhB?usp=sharing)* |

