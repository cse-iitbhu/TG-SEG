from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import AutoConfig, AutoModel, AutoTokenizer


VISION_MODELS = {
    "convnextv2": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "swinv2": "swinv2_tiny_window8_256.ms_in1k",
}

TEXT_MODELS = {
    "roberta": "FacebookAI/roberta-base",
    "bmb": "thomas-sounack/BioClinical-ModernBERT-base",
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
}


class ResNet50Backbone(nn.Module):
    """The same C2-C5 ResNet-50 feature interface used by the main code."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class TimmFeatureBackbone(nn.Module):
    """Expose any timm hierarchical encoder as four NCHW feature levels."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: int = 512,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "ConvNeXtV2 and SwinV2 ablations require `pip install timm`."
            ) from exc

        model_kwargs = {}
        if model_name.startswith("swin"):
            model_kwargs["img_size"] = image_size
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            **model_kwargs,
        )
        self.out_channels = list(self.model.feature_info.channels())
        if len(self.out_channels) != 4:
            raise RuntimeError(
                f"{model_name} returned {len(self.out_channels)} feature levels; expected 4."
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = list(self.model(x))
        normalized = []
        for feature, channels in zip(features, self.out_channels):
            if feature.ndim != 4:
                raise RuntimeError(f"Expected a 4-D feature map, received {feature.shape}.")
            if feature.shape[1] == channels:
                normalized.append(feature)
            elif feature.shape[-1] == channels:
                normalized.append(feature.permute(0, 3, 1, 2).contiguous())
            else:
                raise RuntimeError(
                    f"Cannot locate the channel dimension ({channels}) in {feature.shape}."
                )
        return normalized


class FPNProjection(nn.Module):
    def __init__(self, in_channels: Sequence[int], hidden_dim: int):
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, hidden_dim, kernel_size=1)
            for channels in in_channels
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [projection(feature) for projection, feature in zip(self.projections, features)]


class HuggingFaceTextEncoder(nn.Module):
    """Shared word/sentence feature interface for every Table 5 text model."""

    def __init__(
        self,
        model_id: str,
        hidden_dim: int = 256,
        freeze: bool = False,
        pretrained: bool = True,
        max_length: int = 128,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if pretrained:
            self.model = AutoModel.from_pretrained(model_id)
        else:
            model_config = AutoConfig.from_pretrained(model_id)
            self.model = AutoModel.from_config(model_config)

        encoder_width = int(self.model.config.hidden_size)
        self.resizer = nn.Linear(encoder_width, hidden_dim)
        self.max_length = max_length

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def forward(
        self, captions: Sequence[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        output = self.model(**tokens)

        token_features = output.last_hidden_state
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            valid = tokens.attention_mask.unsqueeze(-1).to(token_features.dtype)
            pooled = (token_features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        word_features = self.resizer(token_features)
        sentence_features = self.resizer(pooled)
        padding_mask = tokens.attention_mask == 0
        return word_features, sentence_features, padding_mask


def build_vision_encoder(
    name: str,
    pretrained: bool = True,
    image_size: int = 512,
) -> nn.Module:
    if name == "resnet50":
        return ResNet50Backbone(pretrained=pretrained)
    try:
        model_name = VISION_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported vision encoder: {name}") from exc
    return TimmFeatureBackbone(
        model_name,
        pretrained=pretrained,
        image_size=image_size,
    )


def build_text_encoder(
    name: str,
    hidden_dim: int,
    freeze: bool,
    pretrained: bool = True,
) -> HuggingFaceTextEncoder:
    try:
        model_id = TEXT_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported text encoder: {name}") from exc
    return HuggingFaceTextEncoder(
        model_id=model_id,
        hidden_dim=hidden_dim,
        freeze=freeze,
        pretrained=pretrained,
    )
