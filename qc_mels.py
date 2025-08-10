#!/usr/bin/env python3
import os
import csv
import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def mel_qc(mel, expect_bins=None, value_range=(-20.0, 10.0)):
    rep = {}
    if mel.ndim != 2:
        rep['ndim_not_2'] = True
        return True, rep

    T, M = mel.shape
    if T < M:
        mel = mel.T
        T, M = mel.shape
        rep['transposed'] = True

    if not np.isfinite(mel).all():
        rep['reason'] = 'nan_or_inf'
        return True, rep

    vmin, vmax = float(mel.min()), float(mel.max())
    rep['range'] = vmax - vmin
    lo, hi = value_range
    if vmin < lo - 5 or vmax > hi + 5:
        return True, {'reason': 'values_out_of_expected_range'}

    # spectral flux (L2 diff over time, averaged)
    if T > 1:
        d = np.diff(mel, axis=0)
        flux = float(np.mean(np.sqrt((d**2).sum(axis=1) / (M + 1e-8))))
    else:
        flux = 0.0

    # per-bin temporal std (median over bins)
    bin_time_std = np.std(mel, axis=0)
    med_time_std = float(np.median(bin_time_std))

    # gradient anisotropy time vs freq
    dt = float(np.mean(np.abs(np.diff(mel, axis=0)))) if T > 1 else 0.0
    df = float(np.mean(np.abs(np.diff(mel, axis=1)))) if M > 1 else 0.0
    ratio_t_over_f = (dt + 1e-8) / (df + 1e-8)

    # centroid variance over time
    w = np.exp(mel - np.max(mel, axis=1, keepdims=True))
    idx = np.arange(M, dtype=np.float32)[None, :]
    cent = (w * idx).sum(axis=1) / (w.sum(axis=1) + 1e-8)
    cent_var = float(np.var(cent))

    # rank-1 energy via SVD
    X = mel - mel.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(X, compute_uv=False)
        rank1_energy = float((s[0]**2) / (np.sum(s**2) + 1e-8))
    except np.linalg.LinAlgError:
        rank1_energy = 1.0  # fail safe → flag as bad

    rep.update(dict(
        flux=flux,
        med_time_std=med_time_std,
        ratio_t_over_f=ratio_t_over_f,
        cent_var=cent_var,
        rank1_energy=rank1_energy,
        value_min=vmin,
        value_max=vmax,
    ))

    reasons = []

    if rank1_energy > 0.985:
        reasons.append('rank1_like')

    if flux < 0.03 and med_time_std < 0.12:
        reasons.append('very_low_temporal_change')

    if ratio_t_over_f < 0.15:
        reasons.append('time_grad_tiny_vs_freq')

    if cent_var < 5.0:
        reasons.append('centroid_static')

    if rep['range'] < 0.6:
        reasons.append('nearly_constant_values')

    bad = len(reasons) > 1
    rep['reasons'] = reasons
    return bad, rep


def scan_and_move(src_dir, bad_dir, csv_path, expect_bins=None, value_low=-20.0, value_high=10.0, mirror=False):
    src_dir = Path(src_dir)
    bad_dir = Path(bad_dir)
    csv_path = Path(csv_path)
    bad_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    bad = 0

    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'rel_path',
            'abs_path',
            'moved_to',
            'reasons',
            'rank1_energy',
            'flux',
            'median_time_std',
            'ratio_time_over_freq',
            'centroid_var',
            'value_min',
            'value_max',
            'range'
        ]
        writer.writerow(header)

        for npy_file in tqdm(src_dir.rglob('*.npy'), desc="Scanning", unit="file"):
            total += 1
            try:
                mel = np.load(npy_file)
            except Exception as e:
                # treat unreadable as bad
                rel = npy_file.relative_to(src_dir)
                dst = bad_dir / (rel if mirror else rel.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(npy_file), str(dst))
                writer.writerow([str(rel), str(npy_file), str(dst), 'load_error:' + str(e),
                                 '', '', '', '', '', '', '', ''])
                bad += 1
                continue

            is_bad, rep = mel_qc(mel, expect_bins=expect_bins, value_range=(value_low, value_high))

            if is_bad:
                rel = npy_file.relative_to(src_dir)
                dst = bad_dir / (rel if mirror else rel.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(npy_file), str(dst))

                reasons = ';'.join(rep.get('reasons', [rep.get('reason', 'unknown')]))
                writer.writerow([
                    str(rel),
                    str(npy_file),
                    str(dst),
                    reasons,
                    rep.get('rank1_energy', ''),
                    rep.get('flux', ''),
                    rep.get('med_time_std', ''),
                    rep.get('ratio_t_over_f', ''),
                    rep.get('cent_var', ''),
                    rep.get('value_min', ''),
                    rep.get('value_max', ''),
                    rep.get('range', ''),
                ])
                bad += 1

    print(f'Done. Checked {total} files. Flagged & moved {bad}. CSV -> {csv_path}')


def main():
    p = argparse.ArgumentParser(description='Scan .npy mels, move bad ones, log reasons to CSV')
    p.add_argument('src', help='Source directory (scanned recursively)')
    p.add_argument('dst', help='Destination directory for bad files')
    p.add_argument('--csv', default='bad_mels.csv', help='CSV output path (default: bad_mels.csv)')
    p.add_argument('--expect-bins', type=int, default=None, help='Expected mel bins (e.g., 128). If set, mismatch -> bad')
    p.add_argument('--val-min', type=float, default=-20.0, help='Expected lower bound of values')
    p.add_argument('--val-max', type=float, default=10.0, help='Expected upper bound of values')
    p.add_argument('--mirror', action='store_true', help='Mirror the source subfolder structure in dst')
    args = p.parse_args()

    scan_and_move(
        src_dir=args.src,
        bad_dir=args.dst,
        csv_path=args.csv,
        expect_bins=args.expect_bins,
        value_low=args.val_min,
        value_high=args.val_max,
        mirror=args.mirror
    )


if __name__ == '__main__':
    main()
