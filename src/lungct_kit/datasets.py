"""Read nodule annotation tables and serve nodule-centred patches.

``NoduleDataset`` is a plain map-style dataset (``__len__`` + ``__getitem__``),
so it works directly with ``torch.utils.data.DataLoader`` without lungct-kit
depending on PyTorch.
"""
from __future__ import annotations

import csv
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preprocess import crop_patch, resample, window_hu
from .registry import get_info
from .volume import NPY_SUFFIX, SITK_SUFFIXES, Volume, read_volume, world_to_voxel

IMAGE_SUFFIXES = (*SITK_SUFFIXES, NPY_SUFFIX)


@dataclass(frozen=True)
class Nodule:
    uid: str  # scan (series) identifier
    xyz: tuple[float, float, float]  # world coordinates, mm
    label: int | None = None
    patient: str | None = None  # grouping key for splits
    diameter_mm: float | None = None
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def group(self) -> str:
        """The key to split on: patient if known, otherwise the scan."""
        return self.patient or self.uid


def read_annotations(csv_path: str | Path, dataset: str) -> list[Nodule]:
    """Parse a LUNA16 or LUNA25 annotation CSV into :class:`Nodule` records."""
    cols = get_info(dataset).columns
    if not cols:
        raise ValueError(f"no annotation loader for {dataset!r} yet")
    # Columns stored in dedicated fields; everything else (NoduleID, StudyDate, ...) goes to `extra`.
    known = {cols[k] for k in ("uid", "x", "y", "z", "label", "patient", "diameter") if k in cols}
    nodules: list[Nodule] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [cols[k] for k in ("uid", "x", "y", "z") if cols[k] not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing columns {missing} for {dataset}")
        for row in reader:
            label = row.get(cols.get("label", ""), "")
            diameter = row.get(cols.get("diameter", ""), "")
            nodules.append(Nodule(
                uid=row[cols["uid"]].strip(),
                xyz=(float(row[cols["x"]]), float(row[cols["y"]]), float(row[cols["z"]])),
                label=int(float(label)) if label not in ("", None) else None,
                patient=(row.get(cols.get("patient", ""), "") or "").strip() or None,
                diameter_mm=float(diameter) if diameter not in ("", None) else None,
                extra={k: v for k, v in row.items() if k not in known},
            ))
    return nodules


def index_images(root: str | Path, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> dict[str, Path]:
    """Find every image under ``root`` and map its series UID (file stem) to its path.

    Works for LUNA16 ``subsetN/<uid>.mhd`` and LUNA25 ``<uid>.mha`` layouts alike.
    """
    suffixes = tuple(s.lower() for s in suffixes)
    found: dict[str, Path] = {}
    for p in Path(root).rglob("*"):
        name = p.name.lower()
        for s in suffixes:
            if name.endswith(s) and p.is_file():
                uid = p.name[: -len(s)]
                if uid in found:
                    raise ValueError(f"series {uid} found twice: {found[uid]} and {p}")
                found[uid] = p
                break
    return found


def luna16_subset(path: Path) -> int | None:
    """Return the official LUNA16 fold (0-9) from a path like ``.../subset3/<uid>.mhd``."""
    for part in path.parts:
        if part.startswith("subset") and part[6:].isdigit():
            return int(part[6:])
    return None


class NoduleDataset:
    """Nodule-centred 3D patches, resampled and windowed, ready for a CNN.

    Each item is a dict::

        {"image": float32 array (1, D, H, W) in [0, 1], "label": int or None,
         "uid": str, "group": str, "center_voxel": (z, y, x)}

    Recently used scans are kept in a small cache, so iterate in scan order
    (the default ordering groups nodules by scan) for speed.
    """

    def __init__(
        self,
        nodules: list[Nodule],
        images: dict[str, Path],
        patch_size: int | tuple[int, int, int] = 64,
        spacing: float | tuple[float, float, float] | None = 1.0,
        window: tuple[float, float] | None = (-1000.0, 400.0),
        transform: Callable[[dict], dict] | None = None,
        cache_size: int = 4,
        drop_missing: bool = True,
    ) -> None:
        missing = [n for n in nodules if n.uid not in images]
        if missing and not drop_missing:
            raise FileNotFoundError(f"{len(missing)} nodules have no image, e.g. {missing[0].uid}")
        self.missing = missing
        self.nodules = sorted((n for n in nodules if n.uid in images), key=lambda n: n.uid)
        self.images = images
        self.patch_size = patch_size
        self.spacing = spacing
        self.window = window
        self.transform = transform
        self._cache: OrderedDict[str, Volume] = OrderedDict()
        self._cache_size = max(1, cache_size)

    def __len__(self) -> int:
        return len(self.nodules)

    @property
    def groups(self) -> list[str]:
        return [n.group for n in self.nodules]

    @property
    def labels(self) -> list[int | None]:
        return [n.label for n in self.nodules]

    def volume(self, uid: str) -> Volume:
        if uid in self._cache:
            self._cache.move_to_end(uid)
            return self._cache[uid]
        vol = read_volume(self.images[uid])
        if self.spacing is not None:
            vol = resample(vol, self.spacing)
        self._cache[uid] = vol
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vol

    def __getitem__(self, i: int) -> dict:
        n = self.nodules[i]
        vol = self.volume(n.uid)
        center = world_to_voxel(n.xyz, vol)
        patch = crop_patch(vol.array, center, self.patch_size)
        image = window_hu(patch, *self.window) if self.window else patch.astype(np.float32)
        item = {
            "image": image[None],
            "label": n.label,
            "uid": n.uid,
            "group": n.group,
            "center_voxel": tuple(float(c) for c in center),
        }
        return self.transform(item) if self.transform else item

    def subset(self, indices) -> NoduleDataset:
        """A new dataset with only the given indices (e.g. one fold)."""
        ds = NoduleDataset.__new__(NoduleDataset)
        ds.__dict__.update(self.__dict__)
        ds.nodules = [self.nodules[i] for i in indices]
        ds._cache = OrderedDict()
        return ds
