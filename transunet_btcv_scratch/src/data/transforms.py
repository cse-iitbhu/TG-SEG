from __future__ import annotations

from typing import Sequence

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)


def get_train_transforms(
    spacing: Sequence[float],
    crop_size: Sequence[int],
    num_samples: int,
):
    return Compose(
        [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            Spacingd(keys=['image', 'label'], pixdim=spacing, mode=('bilinear', 'nearest')),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=-125,
                a_max=275,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            RandCropByPosNegLabeld(
                keys=['image', 'label'],
                label_key='label',
                spatial_size=tuple(crop_size),
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key='image',
                image_threshold=0.0,
            ),
            RandFlipd(keys=['image', 'label'], spatial_axis=0, prob=0.5),
            RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=0.5),
            RandFlipd(keys=['image', 'label'], spatial_axis=2, prob=0.5),
            RandAffined(
                keys=['image', 'label'],
                prob=0.25,
                rotate_range=(0.15, 0.15, 0.15),
                scale_range=(0.1, 0.1, 0.1),
                mode=('bilinear', 'nearest'),
                padding_mode='border',
            ),
            RandGaussianNoised(keys=['image'], prob=0.15, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=['image'], prob=0.1, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.15),
            RandAdjustContrastd(keys=['image'], prob=0.15, gamma=(0.7, 1.5)),
            EnsureTyped(keys=['image', 'label']),
        ]
    )


def get_val_transforms(spacing: Sequence[float]):
    return Compose(
        [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            Spacingd(keys=['image', 'label'], pixdim=spacing, mode=('bilinear', 'nearest')),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=-125,
                a_max=275,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=['image', 'label']),
        ]
    )


def get_test_transforms(spacing: Sequence[float]):
    return Compose(
        [
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            Orientationd(keys=['image'], axcodes='RAS'),
            Spacingd(keys=['image'], pixdim=spacing, mode=('bilinear',)),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=-125,
                a_max=275,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=['image']),
        ]
    )
