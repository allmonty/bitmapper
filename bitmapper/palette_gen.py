"""Automatic palette generation from an image's own colors.

Two hand-rolled algorithms (no scikit-learn dependency, so the logic is easy
to reason about and port later): median cut and k-means.
"""
from __future__ import annotations

import numpy as np


def _bucket_priority(bucket: np.ndarray) -> float:
    if len(bucket) <= 1:
        return -1.0
    ranges = bucket.max(axis=0) - bucket.min(axis=0)
    return float(ranges.max()) * len(bucket)


def _pad_palette(palette: np.ndarray, n_colors: int) -> np.ndarray:
    if len(palette) == 0:
        return np.zeros((n_colors, 3), dtype=np.uint8)
    if len(palette) < n_colors:
        pad = np.tile(palette[-1], (n_colors - len(palette), 1))
        palette = np.vstack([palette, pad])
    return palette[:n_colors]


def median_cut(pixels: np.ndarray, n_colors: int) -> np.ndarray:
    """Classic median-cut color quantization.

    Repeatedly splits the bucket with the largest (range * population) along
    its widest channel, until there are ``n_colors`` buckets (or no bucket
    can be split further, e.g. a single-color image).
    """
    if n_colors < 1:
        raise ValueError("n_colors must be >= 1")

    flat = pixels.reshape(-1, 3).astype(np.float64)
    unique_pixels = np.unique(flat, axis=0)
    buckets = [unique_pixels]

    while len(buckets) < n_colors:
        buckets.sort(key=_bucket_priority)
        bucket = buckets[-1]
        if len(bucket) <= 1:
            break
        buckets.pop()

        channel = int(np.argmax(bucket.max(axis=0) - bucket.min(axis=0)))
        bucket = bucket[bucket[:, channel].argsort()]
        mid = len(bucket) // 2
        buckets.append(bucket[:mid])
        buckets.append(bucket[mid:])

    palette = np.array([bucket.mean(axis=0) for bucket in buckets])
    palette = np.clip(palette, 0, 255).astype(np.uint8)
    return _pad_palette(palette, n_colors)


def kmeans(pixels: np.ndarray, n_colors: int, iterations: int = 10, seed: int = 0) -> np.ndarray:
    """Simple Lloyd's-algorithm k-means quantization."""
    if n_colors < 1:
        raise ValueError("n_colors must be >= 1")

    rng = np.random.default_rng(seed)
    flat = pixels.reshape(-1, 3).astype(np.float64)
    unique_pixels = np.unique(flat, axis=0)

    k = min(n_colors, len(unique_pixels))
    chosen = rng.choice(len(unique_pixels), size=k, replace=False)
    centers = unique_pixels[chosen].copy()

    for _ in range(iterations):
        dists = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centers[i] = flat[mask].mean(axis=0)
        centers = new_centers

    palette = np.clip(centers, 0, 255).astype(np.uint8)
    return _pad_palette(palette, n_colors)


def generate_palette(pixels: np.ndarray, n_colors: int, algorithm: str = "median_cut") -> np.ndarray:
    if algorithm == "median_cut":
        return median_cut(pixels, n_colors)
    if algorithm == "kmeans":
        return kmeans(pixels, n_colors)
    raise ValueError(f"unknown palette algorithm: {algorithm!r}")
