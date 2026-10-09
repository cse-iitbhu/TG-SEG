from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that changes between the paper's ablation variants."""

    name: str
    vision_encoder: str = "resnet50"
    text_encoder: str = "roberta"
    use_presence: bool = True
    hidden_dim: int = 256
    num_queries: int = 5
    use_propagation: bool = True
    presence_tau: float = 0.5
    lambda_presence: float = 1.0
    lambda_empty: float = 0.0
    freeze_text_encoder: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# Table 5: all variants use the complete presence-aware TG-SEG model.
TABLE5_EXPERIMENTS = (
    "resnet50_roberta",
    "resnet50_bmb",
    "resnet50_pubmedbert",
    "convnextv2_roberta",
    "swinv2_roberta",
)

# Tables 3 and 4: same encoders and propagation, presence is the only switch.
PRESENCE_EXPERIMENTS = (
    "resnet50_roberta_presence",
    "resnet50_roberta_no_presence",
)


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "resnet50_roberta": ExperimentConfig(
        name="resnet50_roberta",
        vision_encoder="resnet50",
        text_encoder="roberta",
        use_presence=True,
    ),
    "resnet50_bmb": ExperimentConfig(
        name="resnet50_bmb",
        vision_encoder="resnet50",
        text_encoder="bmb",
        use_presence=True,
    ),
    "resnet50_pubmedbert": ExperimentConfig(
        name="resnet50_pubmedbert",
        vision_encoder="resnet50",
        text_encoder="pubmedbert",
        use_presence=True,
    ),
    "convnextv2_roberta": ExperimentConfig(
        name="convnextv2_roberta",
        vision_encoder="convnextv2",
        text_encoder="roberta",
        use_presence=True,
    ),
    "swinv2_roberta": ExperimentConfig(
        name="swinv2_roberta",
        vision_encoder="swinv2",
        text_encoder="roberta",
        use_presence=True,
    ),
    "resnet50_roberta_presence": ExperimentConfig(
        name="resnet50_roberta_presence",
        vision_encoder="resnet50",
        text_encoder="roberta",
        use_presence=True,
    ),
    "resnet50_roberta_no_presence": ExperimentConfig(
        name="resnet50_roberta_no_presence",
        vision_encoder="resnet50",
        text_encoder="roberta",
        use_presence=False,
    ),
}


def get_experiment(name: str, **overrides) -> ExperimentConfig:
    try:
        config = EXPERIMENTS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment '{name}'. Available: {choices}") from exc
    return replace(config, **overrides) if overrides else config
