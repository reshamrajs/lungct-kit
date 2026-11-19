"""lungct-kit: leakage-safe loaders, preprocessing and splits for public lung CT datasets."""
from .datasets import Nodule, NoduleDataset, index_images, luna16_subset, read_annotations
from .preprocess import LUNG_WINDOW, crop_patch, resample, window_hu
from .registry import DATASETS, DatasetInfo, get_info
from .splits import LeakageError, check_no_leakage, group_kfold
from .volume import Volume, read_volume, voxel_to_world, world_to_voxel, write_volume

__version__ = "0.1.0"


def load(dataset: str, annotations: str, images: str, **kwargs) -> NoduleDataset:
    """One-line setup: read the annotation CSV, index the images, build the dataset.

    >>> ds = load("luna25", "LUNA25_Public_Training_Development_Data.csv", "luna25_images/")
    >>> folds = group_kfold(ds.groups, ds.labels, n_folds=5)
    """
    return NoduleDataset(read_annotations(annotations, dataset), index_images(images), **kwargs)


__all__ = [
    "DATASETS",
    "LUNG_WINDOW",
    "DatasetInfo",
    "LeakageError",
    "Nodule",
    "NoduleDataset",
    "Volume",
    "check_no_leakage",
    "crop_patch",
    "get_info",
    "group_kfold",
    "index_images",
    "load",
    "luna16_subset",
    "read_annotations",
    "read_volume",
    "resample",
    "voxel_to_world",
    "window_hu",
    "world_to_voxel",
    "write_volume",
]
