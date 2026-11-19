"""Standard CT preprocessing: HU windowing, isotropic resampling, patch cropping."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from .volume import Volume

AIR_HU = -1024
# Common lung window used by most nodule papers.
LUNG_WINDOW = (-1000.0, 400.0)


def window_hu(array: np.ndarray, low: float = LUNG_WINDOW[0], high: float = LUNG_WINDOW[1]) -> np.ndarray:
    """Clip to ``[low, high]`` HU and rescale to ``[0, 1]`` as float32."""
    if high <= low:
        raise ValueError("high must be greater than low")
    out = np.clip(array.astype(np.float32), low, high)
    return (out - low) / (high - low)


def resample(volume: Volume, new_spacing: float | Sequence[float] = 1.0, order: int = 1) -> Volume:
    """Resample to a new voxel spacing in mm (``1.0`` = 1 mm isotropic).

    The output spacing is adjusted slightly so the physical extent is preserved
    exactly with a whole number of voxels. The origin (centre of voxel 0) is kept.
    """
    target = np.broadcast_to(np.asarray(new_spacing, dtype=float), (3,))
    if np.any(target <= 0):
        raise ValueError("new_spacing must be positive")
    old = np.asarray(volume.spacing)
    shape = np.asarray(volume.shape)
    new_shape = np.maximum(np.round(shape * old / target), 1).astype(int)
    zoom = new_shape / shape
    arr = ndimage.zoom(volume.array.astype(np.float32), zoom, order=order, mode="nearest", grid_mode=True)
    if np.issubdtype(volume.array.dtype, np.integer):
        arr = np.round(arr).astype(volume.array.dtype)
    return Volume(arr, tuple(old * shape / new_shape), volume.origin, volume.direction)


def crop_patch(
    array: np.ndarray,
    center_zyx: Sequence[float],
    size_zyx: int | Sequence[int],
    pad_value: float = AIR_HU,
) -> np.ndarray:
    """Cut a box of ``size_zyx`` voxels centred on ``center_zyx``.

    Parts of the box outside the scan are filled with ``pad_value`` (air by
    default), so nodules near the edge still give full-size patches.
    """
    size = np.broadcast_to(np.asarray(size_zyx, dtype=int), (3,))
    if np.any(size <= 0):
        raise ValueError("patch size must be positive")
    center = np.round(np.asarray(center_zyx, dtype=float)).astype(int)
    start = center - size // 2
    out = np.full(tuple(size), pad_value, dtype=array.dtype)

    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(start + size, array.shape)
    if np.any(src_hi <= src_lo):
        return out  # box is completely outside the scan
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    out[tuple(slice(a, b) for a, b in zip(dst_lo, dst_hi, strict=False))] = array[
        tuple(slice(a, b) for a, b in zip(src_lo, src_hi, strict=False))
    ]
    return out
