"""Color helpers shared across the engine stages."""
from __future__ import annotations

import numpy as np


def luminance(colors: np.ndarray) -> np.ndarray:
    """Rec. 601 luma, as ``(r*0.299 + g*0.587) + b*0.114`` in float64 (the
    Dart port computes it in the same order, so results match exactly)."""
    c = colors.astype(np.float64)
    return c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114
