"""Training configuration (single place to change horizons, models, splits...)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.evaluation.splits import SplitConfig
from src.features.pipeline import FeatureConfig
from src.models.registry import DEFAULT_MODELS
from src.peaks.detector import PeakConfig

# Complexity ranking used for tie-breaking: simpler models win ties.
COMPLEXITY = {"baseline": 0, "statistical": 1, "ml": 2}


@dataclass
class TrainingConfig:
    source: str = "synthetic"
    # Horizon buckets in hours. 1h, 6h, 24h and 7 days (168h) as required.
    horizons: tuple[int, ...] = (1, 6, 24, 168)
    models: tuple[str, ...] = DEFAULT_MODELS
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    peaks: PeakConfig = field(default_factory=PeakConfig)
    interval_coverage: float = 0.90
    # A simpler model is preferred when its CV MAE is within this relative margin of the best.
    selection_tolerance: float = 0.01
    track_mlflow: bool = True

    def to_dict(self) -> dict:
        return asdict(self)
