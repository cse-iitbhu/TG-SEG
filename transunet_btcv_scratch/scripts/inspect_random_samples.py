from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from src.utils.io import load_json


def pick_slice(image: np.ndarray, label: np.ndarray | None):
    if label is not None and label.max() > 0:
        sums = label.reshape(label.shape[0], -1).sum(axis=1)
        z = int(np.argmax(sums))
    else:
        z = image.shape[0] // 2
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_cases', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    manifest = load_json(args.manifest)
    cases = manifest['training']
    chosen = random.sample(cases, k=min(args.num_cases, len(cases)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in chosen:
        image = sitk.GetArrayFromImage(sitk.ReadImage(item['image']))
        label = sitk.GetArrayFromImage(sitk.ReadImage(item['label'])) if 'label' in item else None
        z = pick_slice(image, label)
        img_slice = image[z]

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img_slice, cmap='gray')
        plt.title(f"Case {item['case_id']} | slice {z}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_slice, cmap='gray')
        if label is not None:
            # plt.imshow(label[z], alpha=0.35, cmap='tab20')
            masked_label = np.ma.masked_where(label[z] == 0, label[z])
            plt.imshow(masked_label, alpha=0.35, cmap='tab20')
        plt.title('Overlay')
        plt.axis('off')

        out_path = output_dir / f"case_{item['case_id']}_slice_{z}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
