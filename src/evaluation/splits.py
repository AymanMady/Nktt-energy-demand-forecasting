"""Chronological validation splits.

WHY NOT A RANDOM SPLIT?
=======================
Hourly demand is strongly autocorrelated (lag-1 autocorrelation ~0.93 on the
prototype data). With a random ``train_test_split``, the hours surrounding
every test hour sit in the training set: the model would be "predicting"
10:00 after having seen 09:00 and 11:00, and lag/rolling features of test rows
would overlap training targets. Scores would be optimistic and meaningless
for a system that must forecast the *future*. The trend (demand grows every
year) makes it worse: a random split lets the model see future demand levels.

Scheme used here
================
::

    |------------------ development -------------------|------ test ------|
    |== train ==>| fold 1 |                               (last 92 days, touched once)
    |===== train =====>| fold 2 |
    |========= train =========>| fold 3 |
    |============= train =============>| fold 4 |

* Expanding-window folds of ``fold_days`` each, immediately before the test period.
  They are used for model *selection* and to collect out-of-sample residuals
  (prediction intervals).
* The final test period is only used to *report* performance of models trained
  on all development data. It never influences choices.
* Purge: for a horizon-``H`` model, training targets must be observed at the
  forecast origin, so training rows end at ``valid_start - H`` hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    test_days: int = 92
    # 6 x 60 days = one full year before the test period, so that model selection and
    # interval calibration see every season (hot/humid summer included).
    n_folds: int = 6
    fold_days: int = 60


@dataclass(frozen=True)
class Fold:
    name: str
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp  # inclusive

    def train_end(self, horizon: int) -> pd.Timestamp:
        """Last training target timestamp allowed for a horizon-``H`` model (purge)."""
        return self.valid_start - pd.Timedelta(hours=horizon)

    def train_mask(self, index: pd.DatetimeIndex, horizon: int):
        return index <= self.train_end(horizon)

    def valid_mask(self, index: pd.DatetimeIndex):
        return (index >= self.valid_start) & (index <= self.valid_end)


def make_splits(
    index: pd.DatetimeIndex, config: SplitConfig | None = None
) -> tuple[list[Fold], Fold]:
    """Return (cross-validation folds, final test fold) for an hourly index."""
    config = config or SplitConfig()
    end = index.max()
    # Test period = last `test_days` whole days (aligned on midnight).
    test_start = (end + pd.Timedelta(hours=1)).floor("D") - pd.Timedelta(days=config.test_days)
    if test_start <= index.min():
        raise ValueError("Not enough data for the requested test period")
    test = Fold("test", test_start, end)

    folds: list[Fold] = []
    for k in range(config.n_folds):
        start = test_start - pd.Timedelta(days=config.fold_days * (config.n_folds - k))
        stop = start + pd.Timedelta(days=config.fold_days) - pd.Timedelta(hours=1)
        if start <= index.min():
            raise ValueError("Not enough data for the requested cross-validation folds")
        folds.append(Fold(f"fold_{k + 1}", start, stop))
    return folds, test
