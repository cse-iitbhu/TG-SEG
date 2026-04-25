from __future__ import annotations

from typing import Sequence

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    Spacingd,
    SpatialPadd,
)


def get_train_transforms(
    spacing: Sequence[float],
    crop_size: Sequence[int],
    num_samples: int,
):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=tuple(crop_size)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(crop_size),
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0.0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandAffined(
                keys=["image", "label"],
                prob=0.20,
                rotate_range=(0.10, 0.10, 0.10),
                scale_range=(0.10, 0.10, 0.10),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["image"], prob=0.10, mean=0.0, std=0.01),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.10,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.15),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms(spacing: Sequence[float]):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_test_transforms(spacing: Sequence[float]):
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )
    
    
    
    
    
    
# from __future__ import annotations

# from typing import Sequence

# from monai.transforms import (
#     Compose,
#     CropForegroundd,
#     EnsureChannelFirstd,
#     EnsureTyped,
#     LoadImaged,
#     NormalizeIntensityd,
#     Orientationd,
#     RandAffined,
#     RandCropByPosNegLabeld,
#     RandFlipd,
#     RandGaussianNoised,
#     RandGaussianSmoothd,
#     RandScaleIntensityd,
#     Spacingd,
# )


# def get_train_transforms(
#     spacing: Sequence[float],
#     crop_size: Sequence[int],
#     num_samples: int,
# ):
#     return Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
#             NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#             CropForegroundd(keys=["image", "label"], source_key="image"),
#             RandCropByPosNegLabeld(
#                 keys=["image", "label"],
#                 label_key="label",
#                 spatial_size=tuple(crop_size),
#                 pos=1,
#                 neg=1,
#                 num_samples=num_samples,
#                 image_key="image",
#                 image_threshold=0.0,
#             ),
#             RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
#             RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
#             RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
#             RandAffined(
#                 keys=["image", "label"],
#                 prob=0.20,
#                 rotate_range=(0.10, 0.10, 0.10),
#                 scale_range=(0.10, 0.10, 0.10),
#                 mode=("bilinear", "nearest"),
#                 padding_mode="border",
#             ),
#             RandGaussianNoised(keys=["image"], prob=0.10, mean=0.0, std=0.01),
#             RandGaussianSmoothd(
#                 keys=["image"],
#                 prob=0.10,
#                 sigma_x=(0.5, 1.0),
#                 sigma_y=(0.5, 1.0),
#                 sigma_z=(0.5, 1.0),
#             ),
#             RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.15),
#             EnsureTyped(keys=["image", "label"]),
#         ]
#     )


# def get_val_transforms(spacing: Sequence[float]):
#     return Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
#             NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#             EnsureTyped(keys=["image", "label"]),
#         ]
#     )


# def get_test_transforms(spacing: Sequence[float]):
#     return Compose(
#         [
#             LoadImaged(keys=["image"]),
#             EnsureChannelFirstd(keys=["image"]),
#             Orientationd(keys=["image"], axcodes="RAS"),
#             Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear",)),
#             NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
#             EnsureTyped(keys=["image"]),
#         ]
#     )