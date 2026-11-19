# lungct-kit

[![Docs](https://img.shields.io/badge/%F0%9F%93%96_Open_the_docs-2563eb?style=for-the-badge)](https://reshamrajs.github.io/lungct-kit/)
[![All tools](https://img.shields.io/badge/All_open_tools-555?style=for-the-badge)](https://reshamrajs.github.io/open-tools/)
[![Tests](https://github.com/reshamrajs/lungct-kit/actions/workflows/tests.yml/badge.svg)](https://github.com/reshamrajs/lungct-kit/actions/workflows/tests.yml)

**Leakage-safe loaders, preprocessing and splits for public lung CT nodule datasets.**

Every new lung nodule project starts the same way: weeks spent working out file formats,
world-to-voxel coordinates, resampling, patch cropping, and how to split the data. And many
published results are inflated because the same patient ended up in both train and test.
lungct-kit does this part once, correctly, with tests.

| Dataset | Loader | What it is |
|---|---|---|
| LUNA16 | ✅ | 888 scans, 1,186 nodules; detection and false-positive reduction |
| LUNA25 | ✅ | 2,120 NLST patients, 6,163 annotations; malignancy risk |
| LIDC-IDRI | info only (loader planned) | 1,018 scans with 4-reader ratings |

## Install

```bash
pip install "lungct-kit[io] @ git+https://github.com/reshamrajs/lungct-kit"
```

`[io]` adds SimpleITK for .mha/.mhd/.nii/DICOM files. (A PyPI release is planned.)

From source:

```bash
git clone https://github.com/reshamrajs/lungct-kit && cd lungct-kit
pip install -e ".[dev]" && pytest
```

## Quick start

```python
from lungct_kit import load, group_kfold, check_no_leakage

ds = load(
    "luna25",
    annotations="LUNA25_Public_Training_Development_Data.csv",
    images="luna25_images/",          # any folder layout; files are matched by series UID
    patch_size=64,                    # 64x64x64 voxels
    spacing=1.0,                      # resample to 1 mm isotropic
)

item = ds[0]
item["image"].shape                   # (1, 64, 64, 64), float32 in [0, 1] (lung window)
item["label"], item["group"]          # 0 / 1, and the PatientID

# 5 folds, grouped by patient and stratified by cancer status
for train_idx, test_idx in group_kfold(ds.groups, ds.labels, n_folds=5):
    check_no_leakage(ds.groups, train_idx, test_idx)   # raises if a patient is in both
    train, test = ds.subset(train_idx), ds.subset(test_idx)
```

`NoduleDataset` is a plain map-style dataset, so it plugs straight into PyTorch:

```python
from torch.utils.data import DataLoader
loader = DataLoader(train, batch_size=16, num_workers=4)
```

## Command line

```bash
lungct list                                        # supported datasets
lungct info luna25                                 # licence, size, download links, pitfalls
lungct check luna25 --annotations a.csv --images imgs/   # are all annotated scans present?
lungct split luna25 --annotations a.csv --folds 5 --out splits.json
lungct export luna25 --annotations a.csv --images imgs/ --patch 64 --out patches.npz
```

## Pitfalls it protects you from

- **Patient leakage.** In LUNA25, labels are effectively patient-level (every nodule of a cancer
  patient is malignant), and the number of annotations per patient alone predicts cancer with
  AUC ≈ 0.76. Row-level or nodule-level splits leak. `group_kfold` splits by patient.
- **Coordinate mix-ups.** CSV coordinates are world millimetres `(x, y, z)`; arrays are `[z, y, x]`.
  `world_to_voxel` handles spacing, origin and direction cosines, and is tested against SimpleITK.
- **Variable slice thickness** (1.25 to 2.5 mm in LUNA25). `resample` makes scans isotropic.
- **Nodules near the edge of the scan.** `crop_patch` pads with air instead of crashing or shrinking.

Run `lungct info <dataset>` to see the full list for each dataset.

## Conventions

- Arrays, `spacing` and `origin` are in `(z, y, x)` order, in mm.
- Nodule coordinates (`Nodule.xyz`) are world `(x, y, z)` mm, as in the dataset CSVs.
- Images are matched to annotations by file name: `<SeriesInstanceUID>.mha` (or `.mhd`, `.nii.gz`, `.npy`).

## Contributing

Good first issues: a LIDC-IDRI loader (via pylidc), simple 3D augmentations, a LUNA25 nodule-block
loader, and a docs site. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing the datasets

Please cite the original datasets, not just this tool. `lungct info <dataset>` prints the reference.

## Licence

MIT. The datasets have their own licences (LUNA16 and LUNA25: CC BY 4.0; LIDC-IDRI: CC BY 3.0).
