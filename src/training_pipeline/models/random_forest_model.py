"""Random Forest — tree ensemble. Natively multi-output."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def build_random_forest_model(n_estimators: int = 120, max_depth: int | None = 12):
    """RF supports multi-output Y of shape (n_samples, 72) with no wrapper.
    Trees kept modest (depth 12, leaf 5) so the serialized artifact stays small
    enough for reliable free-tier cloud download."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
