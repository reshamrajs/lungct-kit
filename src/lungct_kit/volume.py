"""CT volume container and reading/writing.

Convention used everywhere in lungct-kit:

* ``array`` is indexed ``[z, y, x]`` (NumPy order, slices first).
* ``spacing`` and ``origin`` are also stored in ``(z, y, x)`` order, in millimetres.
* Nodule coordinates in dataset CSVs are *world* coordinates in ``(x, y, z)`` mm
  (LPS, as written by ITK/SimpleITK). Use :func:`world_to_voxel` to convert.

Formats:

* ``.mha`` / ``.mhd`` / ``.nii`` / ``.nii.gz`` and DICOM folders need SimpleITK
  (``pip install lungct-kit[io]``).
* ``.npy`` with a ``.json`` sidecar (``{"spacing": [z,y,x], "origin": [z,y,x]}``)
  works with NumPy only. Handy for caching and for tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SITK_SUFFIXES = (".mha", ".mhd", ".nii", ".nii.gz", ".nrrd")
NPY_SUFFIX = ".npy"


@dataclass
class Volume:
    """A 3D CT scan in Hounsfield units with its physical geometry."""

    array: np.ndarray
    spacing: tuple[float, float, float]  # (z, y, x) mm
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (z, y, x) mm
    direction: np.ndarray = field(default_factory=lambda: np.eye(3))  # (z, y, x) axes

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError(f"expected a 3D array, got shape {self.array.shape}")
        self.spacing = tuple(float(s) for s in self.spacing)
        self.origin = tuple(float(o) for o in self.origin)
        self.direction = np.asarray(self.direction, dtype=float).reshape(3, 3)
        if any(s <= 0 for s in self.spacing):
            raise ValueError(f"spacing must be positive, got {self.spacing}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.array.shape)  # type: ignore[return-value]

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(n * s for n, s in zip(self.shape, self.spacing, strict=False))  # type: ignore[return-value]


def world_to_voxel(world_xyz, volume: Volume) -> np.ndarray:
    """Convert a world point ``(x, y, z)`` in mm to a fractional voxel index ``(z, y, x)``.

    Handles non-identity direction cosines (oblique or flipped scans).
    """
    world_zyx = np.asarray(world_xyz, dtype=float)[::-1]
    offset = world_zyx - np.asarray(volume.origin)
    # world = origin + direction @ (index * spacing)  =>  index = (direction^-1 @ offset) / spacing
    return np.linalg.solve(volume.direction, offset) / np.asarray(volume.spacing)


def voxel_to_world(index_zyx, volume: Volume) -> np.ndarray:
    """Inverse of :func:`world_to_voxel`; returns world ``(x, y, z)`` in mm."""
    idx = np.asarray(index_zyx, dtype=float)
    world_zyx = np.asarray(volume.origin) + volume.direction @ (idx * np.asarray(volume.spacing))
    return world_zyx[::-1]


def _suffix(path: Path) -> str:
    name = path.name.lower()
    return ".nii.gz" if name.endswith(".nii.gz") else path.suffix.lower()


def _require_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "Reading .mha/.mhd/.nii/DICOM needs SimpleITK: pip install 'lungct-kit[io]'"
        ) from e
    return sitk


def read_volume(path: str | Path) -> Volume:
    """Read a CT volume from a file or a DICOM series folder."""
    path = Path(path)
    if path.is_dir():
        sitk = _require_sitk()
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(str(path))
        if not files:
            raise FileNotFoundError(f"no DICOM series found in {path}")
        reader.SetFileNames(files)
        return _from_sitk(reader.Execute())

    suffix = _suffix(path)
    if suffix == NPY_SUFFIX:
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return Volume(
            array=np.load(path),
            spacing=meta.get("spacing", (1.0, 1.0, 1.0)),
            origin=meta.get("origin", (0.0, 0.0, 0.0)),
            direction=np.asarray(meta.get("direction", np.eye(3).tolist())),
        )
    if suffix in SITK_SUFFIXES:
        sitk = _require_sitk()
        return _from_sitk(sitk.ReadImage(str(path)))
    raise ValueError(f"unsupported file type: {path.name}")


def write_volume(volume: Volume, path: str | Path) -> Path:
    """Write a volume. ``.npy`` writes a JSON sidecar; other suffixes use SimpleITK."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _suffix(path) == NPY_SUFFIX:
        np.save(path, volume.array)
        path.with_suffix(".json").write_text(json.dumps({
            "spacing": list(volume.spacing),
            "origin": list(volume.origin),
            "direction": volume.direction.tolist(),
        }))
        return path
    sitk = _require_sitk()
    img = sitk.GetImageFromArray(volume.array)
    img.SetSpacing(volume.spacing[::-1])
    img.SetOrigin(volume.origin[::-1])
    img.SetDirection(_zyx_to_xyz_direction(volume.direction).flatten().tolist())
    sitk.WriteImage(img, str(path))
    return path


def _xyz_to_zyx_direction(d_xyz: np.ndarray) -> np.ndarray:
    # Reverse both the world-axis rows and the index-axis columns.
    return np.asarray(d_xyz)[::-1, ::-1].copy()


_zyx_to_xyz_direction = _xyz_to_zyx_direction  # the flip is its own inverse


def _from_sitk(img) -> Volume:
    sitk = _require_sitk()
    arr = sitk.GetArrayFromImage(img)
    d_xyz = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)
    return Volume(
        array=arr,
        spacing=tuple(img.GetSpacing())[::-1],
        origin=tuple(img.GetOrigin())[::-1],
        direction=_xyz_to_zyx_direction(d_xyz),
    )
