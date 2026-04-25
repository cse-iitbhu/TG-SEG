from __future__ import annotations

import argparse
import sys
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk

from src.data.constants import BTCV_LABELS, BTCV_ORIGINAL_TO_EIGHT_ORGANS
from src.utils.io import save_json


def extract_case_id(name: str) -> str:
    nums = re.findall(r'\d+', name)
    if not nums:
        raise ValueError(f'Could not find numeric case id in filename: {name}')
    return nums[-1].zfill(4)


def remap_label_image(label_arr: np.ndarray) -> np.ndarray:
    new_label = np.zeros_like(label_arr, dtype=np.uint8)
    for old_id, new_id in BTCV_ORIGINAL_TO_EIGHT_ORGANS.items():
        new_label[label_arr == old_id] = new_id
    return new_label


def pair_training_cases(img_dir: Path, label_dir: Path) -> List[Tuple[Path, Path, str]]:
    images = sorted(img_dir.glob('*.nii.gz'))
    labels = sorted(label_dir.glob('*.nii.gz'))
    image_map = {extract_case_id(p.name): p for p in images}
    label_map = {extract_case_id(p.name): p for p in labels}
    common = sorted(set(image_map) & set(label_map))
    pairs = []
    for case_id in common:
        pairs.append((image_map[case_id], label_map[case_id], case_id))
    return pairs


def create_split(case_ids: List[str], train_count: int, seed: int) -> Dict[str, Dict[str, List[str]]]:
    rng = np.random.default_rng(seed)
    shuffled = case_ids.copy()
    rng.shuffle(shuffled)
    train_ids = sorted(shuffled[:train_count])
    val_ids = sorted(shuffled[train_count:])
    return {
        'official_18_12_seeded': {
            'train': train_ids,
            'val': val_ids,
        },
        'all_train': {
            'train': sorted(case_ids),
            'val': [],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_root', type=str, required=True, help='Path to RawData folder extracted from Synapse RawData.zip')
    parser.add_argument('--output_root', type=str, required=True, help='Path to project data/btcv_processed folder')
    parser.add_argument('--train_count', type=int, default=18)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--copy_images', action='store_true', help='Copy images instead of using original paths in the manifest')
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    training_img_dir = raw_root / 'Training' / 'img'
    training_label_dir = raw_root / 'Training' / 'label'
    testing_img_dir = raw_root / 'Testing' / 'img'

    labels_out_dir = output_root / 'labelsTr_remapped'
    images_out_dir = output_root / 'imagesTr_copied'
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        images_out_dir.mkdir(parents=True, exist_ok=True)

    training_pairs = pair_training_cases(training_img_dir, training_label_dir)
    manifest_training = []
    for image_path, label_path, case_id in training_pairs:
        label_itk = sitk.ReadImage(str(label_path))
        label_arr = sitk.GetArrayFromImage(label_itk)
        new_label_arr = remap_label_image(label_arr)
        new_label_itk = sitk.GetImageFromArray(new_label_arr)
        new_label_itk.CopyInformation(label_itk)
        out_label_path = labels_out_dir / f'case_{case_id}.nii.gz'
        sitk.WriteImage(new_label_itk, str(out_label_path))

        if args.copy_images:
            out_img_path = images_out_dir / f'case_{case_id}.nii.gz'
            if not out_img_path.exists():
                shutil.copy2(image_path, out_img_path)
            manifest_image_path = out_img_path
        else:
            manifest_image_path = image_path.resolve()

        manifest_training.append(
            {
                'case_id': case_id,
                'image': str(manifest_image_path),
                'label': str(out_label_path.resolve()),
            }
        )

    testing_items = []
    if testing_img_dir.exists():
        for image_path in sorted(testing_img_dir.glob('*.nii.gz')):
            case_id = extract_case_id(image_path.name)
            testing_items.append(
                {
                    'case_id': case_id,
                    'image': str(image_path.resolve()),
                }
            )

    case_ids = [x['case_id'] for x in manifest_training]
    split = create_split(case_ids, train_count=args.train_count, seed=args.seed)

    metadata = {
        'dataset_name': 'BTCV_Synapse_8_Organs',
        'num_classes': 9,
        'labels': BTCV_LABELS,
        'training': manifest_training,
        'testing': testing_items,
    }

    save_json(metadata, output_root / 'manifest.json')
    save_json(split, output_root / 'split_btcv.json')

    print(f'Prepared {len(manifest_training)} labeled training cases.')
    print(f'Prepared {len(testing_items)} unlabeled testing cases.')
    print(f'Manifest saved to: {output_root / "manifest.json"}')
    print(f'Splits saved to:   {output_root / "split_btcv.json"}')
    print('Label remap used:')
    for k, v in BTCV_LABELS.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
