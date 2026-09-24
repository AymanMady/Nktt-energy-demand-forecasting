"""Convenience re-export: the implementation lives in :mod:`src.models.statistical.sarima`."""

from src.models.statistical.sarima import SARIMAModel, weekly_fourier

__all__ = ["SARIMAModel", "weekly_fourier"]
