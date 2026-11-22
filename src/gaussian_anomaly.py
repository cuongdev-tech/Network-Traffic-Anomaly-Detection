"""Diagonal Gaussian scoring for offline network-flow anomaly detection."""
from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer


@dataclass
class GaussianAnomalyDetector:
    """Fit independent Gaussian marginals on normal flows."""

    variance_floor: float = 1e-4

    def fit(self, normal):
        transformed = self._transform(normal)
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer_.fit_transform(transformed)
        self.mean_ = values.mean(axis=0)
        self.variance_ = values.var(axis=0).clip(min=self.variance_floor)
        return self

    def score_samples(self, samples):
        """Return mean negative log likelihood per feature. Larger is stranger."""
        values = self.imputer_.transform(self._transform(samples))
        squared_error = (values - self.mean_) ** 2
        return 0.5 * (np.log(2 * np.pi * self.variance_)
                      + squared_error / self.variance_).mean(axis=1)

    @staticmethod
    def _transform(samples):
        values = samples.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        values[~np.isfinite(values)] = np.nan
        return np.sign(values) * np.log1p(np.abs(values))
