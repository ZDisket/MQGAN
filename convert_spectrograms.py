#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
import yaml
import numpy as np
from tqdm import tqdm
import argparse
import multiprocessing
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB

class TorchMelSpectrogramExtractor:
    def __init__(self, spec_config):
        self.config = spec_config
        self.clip_val = 1e-5
        self.transf = torch.nn.Sequential(
            MelSpectrogram(
                sample_rate=self.config['sampling_rate'],
                n_fft=self.config['filter_length'],
                win_length=self.config['win_length'],
                hop_length=self.config['hop_length'],
                n_mels=self.config['n_mel_channels'],
                f_min=self.config['mel_fmin'],
                f_max=self.config['mel_fmax'],
                power = 1.0 # magnitude
            ),
        )

    def get_mel_from_wav(self, wav):
        """wav: (1, T) → mel_norm: (T, n_mels) in [+5, -10]"""
        mel_db   = self.transf(wav)          # (1, n_mels, frames)
        mel_norm = torch.log(torch.clamp(mel_db, min=self.clip_val)) # log-dynamic range compression
        return mel_norm.squeeze(0).T            # (frames, n_mels)

class MelSpectrogramConverter:
    def __init__(self, config):
        self.config = config
        self.extractor = TorchMelSpectrogramExtractor(config['spectrogram'])
        os.makedirs(self.config['io']['output_folder'], exist_ok=True)

    def process_file(self, file_path, output_dir):
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_path = os.path.join(output_dir, f"{base_name}_mel.npy")

        if os.path.isfile(output_file_path):
            return True

        try:
            wav_tensor, sr = torchaudio.load(file_path)
            if sr != self.config['spectrogram']['sampling_rate']:
                resampler = Resample(orig_freq=sr, new_freq=self.config['spectrogram']['sampling_rate'])
                wav_tensor = resampler(wav_tensor)

            duration = wav_tensor.shape[1] / self.config['spectrogram']['sampling_rate']
            if duration < 1.0 or duration > 15.0:
                return False

            mel_spectrogram = self.extractor.get_mel_from_wav(wav_tensor)
            np.save(output_file_path, mel_spectrogram.numpy())
            return True
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

def worker(worker_id, tasks, config):
    converter = MelSpectrogramConverter(config)
    pbar = tqdm(tasks, desc=f"Worker {worker_id}", position=worker_id)
    for file_path, output_dir in pbar:
        os.makedirs(output_dir, exist_ok=True)
        converter.process_file(file_path, output_dir)

def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def validate_config(config):
    required_keys = {
        'io': ['input_folder', 'output_folder', 'audio_extensions'],
        'spectrogram': ['sampling_rate', 'filter_length', 'hop_length', 'win_length', 'n_mel_channels', 'mel_fmin', 'mel_fmax']
    }
    for main_key, sub_keys in required_keys.items():
        if main_key not in config:
            raise ValueError(f"Missing required key in config: '{main_key}'")
        for sub_key in sub_keys:
            if sub_key not in config[main_key]:
                raise ValueError(f"Missing required key in config['{main_key}']: '{sub_key}'")

def main():
    parser = argparse.ArgumentParser(description="Convert audio files to mel spectrograms.")
    parser.add_argument("--config", type=str, default="spec_config.yaml", help="Path to the configuration file.")
    parser.add_argument("--input_folder", type=str, default=None, help="Override the input folder specified in the config file.")
    parser.add_argument("--output_folder", type=str, default=None, help="Override the output folder specified in the config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.input_folder:
        config['io']['input_folder'] = args.input_folder
    if args.output_folder:
        config['io']['output_folder'] = args.output_folder
    
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        exit(1)

    os.makedirs(config['io']['output_folder'], exist_ok=True)

    tasks = []
    audio_exts = tuple(config['io']['audio_extensions'])
    for root, _, files in os.walk(config['io']['input_folder']):
        rel_path = os.path.relpath(root, config['io']['input_folder'])
        output_subfolder = os.path.join(config['io']['output_folder'], rel_path)
        for wav_file in files:
            if wav_file.lower().endswith(audio_exts):
                file_path = os.path.join(root, wav_file)
                tasks.append((file_path, output_subfolder))

    num_workers = multiprocessing.cpu_count()
    task_chunks = chunkify(tasks, num_workers)

    processes = []
    for i, chunk in enumerate(task_chunks):
        p = multiprocessing.Process(target=worker, args=(i, chunk, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
