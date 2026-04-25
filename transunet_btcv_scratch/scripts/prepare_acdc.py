from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to ACDC training folder")
    parser.add_argument("--output", type=str, required=True, help="Output processed folder")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    rng = random.Random(args.seed)

    patient_dirs = sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("patient")])
    if not patient_dirs:
        raise RuntimeError(f"No patient folders found in: {input_root}")

    manifest_items = []
    patient_to_case_ids = {}

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        gt_files = sorted(patient_dir.glob(f"{patient_id}_frame*_gt.nii.gz"))
        patient_to_case_ids[patient_id] = []

        for gt_path in gt_files:
            image_name = gt_path.name.replace("_gt.nii.gz", ".nii.gz")
            image_path = gt_path.with_name(image_name)

            if not image_path.exists():
                print(f"[SKIP] Missing image for label: {gt_path}")
                continue

            case_id = gt_path.name.replace("_gt.nii.gz", "")

            # store paths relative to the processed folder
            rel_image = image_path.resolve().relative_to(output_root.parent.parent.resolve())
            rel_label = gt_path.resolve().relative_to(output_root.parent.parent.resolve())

            manifest_items.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "image": str(rel_image),
                    "label": str(rel_label),
                }
            )
            patient_to_case_ids[patient_id].append(case_id)

    patient_to_case_ids = {k: v for k, v in patient_to_case_ids.items() if len(v) > 0}
    patient_ids = sorted(patient_to_case_ids.keys())
    rng.shuffle(patient_ids)

    if len(patient_ids) == 0:
        raise RuntimeError("No valid ACDC image/label pairs found.")

    n_val_patients = max(1, int(round(len(patient_ids) * args.val_ratio)))
    val_patients = set(patient_ids[:n_val_patients])

    train_case_ids = []
    val_case_ids = []
    all_case_ids = []

    for patient_id in sorted(patient_ids):
        case_ids = sorted(patient_to_case_ids[patient_id])
        all_case_ids.extend(case_ids)
        if patient_id in val_patients:
            val_case_ids.extend(case_ids)
        else:
            train_case_ids.extend(case_ids)

    manifest = {
        "dataset_name": "ACDC_ED_ES",
        "num_classes": 4,
        "labels": {
            "0": "background",
            "1": "rv",
            "2": "myocardium",
            "3": "lv",
        },
        "training": manifest_items,
        "testing": [],
    }

    splits = {
        "patient_80_20_seeded": {
            "train": train_case_ids,
            "val": val_case_ids,
        },
        "all_labeled_as_val": {
            "train": [],
            "val": all_case_ids,
        },
    }

    save_json(output_root / "manifest.json", manifest)
    save_json(output_root / "split_acdc.json", splits)

    print(f"Saved manifest to: {output_root / 'manifest.json'}")
    print(f"Saved splits   to: {output_root / 'split_acdc.json'}")
    print(f"Patients total used: {len(patient_ids)}")
    print(f"Train cases        : {len(train_case_ids)}")
    print(f"Val cases          : {len(val_case_ids)}")
    print(f"All labeled cases  : {len(all_case_ids)}")


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
# from __future__ import annotations

# import argparse
# import json
# import random
# from pathlib import Path


# def save_json(path: Path, obj):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w") as f:
#         json.dump(obj, f, indent=2)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", type=str, required=True, help="Path to ACDC training folder")
#     parser.add_argument("--output", type=str, required=True, help="Output processed folder")
#     parser.add_argument("--val_ratio", type=float, default=0.2)
#     parser.add_argument("--seed", type=int, default=3407)
#     args = parser.parse_args()

#     input_root = Path(args.input)
#     output_root = Path(args.output)
#     rng = random.Random(args.seed)

#     patient_dirs = sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("patient")])
#     if not patient_dirs:
#         raise RuntimeError(f"No patient folders found in: {input_root}")

#     manifest_items = []
#     patient_to_case_ids = {}

#     for patient_dir in patient_dirs:
#         patient_id = patient_dir.name
#         gt_files = sorted(patient_dir.glob(f"{patient_id}_frame*_gt.nii.gz"))

#         if len(gt_files) == 0:
#             print(f"[WARN] No GT files found in {patient_dir}")
#             continue

#         patient_to_case_ids[patient_id] = []

#         for gt_path in gt_files:
#             image_name = gt_path.name.replace("_gt.nii.gz", ".nii.gz")
#             image_path = gt_path.with_name(image_name)
#             if not image_path.exists():
#                 print(f"[WARN] Missing image for label: {gt_path}")
#                 continue

#             frame_id = image_path.stem.replace(".nii", "").split("_")[-1]   # frame01 / frame09
#             case_id = f"{patient_id}_{frame_id}"

#             manifest_items.append(
#                 {
#                     "case_id": case_id,
#                     "patient_id": patient_id,
#                     "frame_id": frame_id,
#                     "image": str(image_path.resolve()),
#                     "label": str(gt_path.resolve()),
#                 }
#             )
#             patient_to_case_ids[patient_id].append(case_id)

#     patient_ids = sorted(patient_to_case_ids.keys())
#     rng.shuffle(patient_ids)

#     n_val_patients = max(1, int(round(len(patient_ids) * args.val_ratio)))
#     val_patients = set(patient_ids[:n_val_patients])
#     train_patients = set(patient_ids[n_val_patients:])

#     train_case_ids = []
#     val_case_ids = []
#     all_case_ids = []

#     for patient_id in sorted(patient_ids):
#         case_ids = sorted(patient_to_case_ids[patient_id])
#         all_case_ids.extend(case_ids)
#         if patient_id in val_patients:
#             val_case_ids.extend(case_ids)
#         else:
#             train_case_ids.extend(case_ids)

#     manifest = {
#         "dataset_name": "ACDC_ED_ES",
#         "num_classes": 4,
#         "labels": {
#             "0": "background",
#             "1": "rv",
#             "2": "myocardium",
#             "3": "lv",
#         },
#         "training": manifest_items,
#         "testing": [],
#     }

#     splits = {
#         "patient_80_20_seeded": {
#             "train": train_case_ids,
#             "val": val_case_ids,
#         },
#         "all_labeled_as_val": {
#             "train": [],
#             "val": all_case_ids,
#         },
#     }

#     save_json(output_root / "manifest.json", manifest)
#     save_json(output_root / "split_acdc.json", splits)

#     print(f"Saved manifest to: {output_root / 'manifest.json'}")
#     print(f"Saved splits   to: {output_root / 'split_acdc.json'}")
#     print(f"Patients total: {len(patient_ids)}")
#     print(f"Train patients: {len(train_patients)}")
#     print(f"Val patients  : {len(val_patients)}")
#     print(f"Train cases   : {len(train_case_ids)}")
#     print(f"Val cases     : {len(val_case_ids)}")
#     print(f"All labeled   : {len(all_case_ids)}")


# if __name__ == "__main__":
#     main()