"""Facts about public lung CT datasets, so users don't have to dig through papers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    full_name: str
    task: str
    licence: str
    homepage: str
    image_format: str
    annotation_file: str
    columns: dict[str, str]
    sources: tuple[str, ...] = ()
    size: str = ""
    loader: bool = True
    pitfalls: tuple[str, ...] = field(default_factory=tuple)
    citation: str = ""


DATASETS: dict[str, DatasetInfo] = {
    "luna16": DatasetInfo(
        name="luna16",
        full_name="LUng Nodule Analysis 2016",
        task="Nodule detection (annotations.csv) and false-positive reduction (candidates.csv)",
        licence="CC BY 4.0 (derived from LIDC-IDRI)",
        homepage="https://luna16.grand-challenge.org",
        image_format="MetaImage .mhd + .raw, split into subset0 ... subset9",
        annotation_file="annotations.csv (or candidates_V2.csv for candidates)",
        columns={
            "uid": "seriesuid",
            "x": "coordX",
            "y": "coordY",
            "z": "coordZ",
            "diameter": "diameter_mm",
            "label": "class",
        },
        size="888 CT scans, 1,186 nodules >= 3 mm accepted by >= 3 of 4 radiologists",
        pitfalls=(
            "The 10 subsets are the official cross-validation folds; use them for comparable results.",
            "annotations.csv has no label column (all rows are nodules); candidates.csv has 'class'.",
            "Patient IDs are not given; one scan per patient is assumed, so seriesuid is the group.",
        ),
        citation="Setio et al., Medical Image Analysis 42 (2017) 1-13.",
    ),
    "luna25": DatasetInfo(
        name="luna25",
        full_name="LUNA25: Lung Nodule Malignancy Risk Estimation Challenge 2025",
        task="Binary malignancy risk of a given nodule (CT + nodule centre -> P(malignant))",
        licence="CC BY 4.0",
        homepage="https://luna25.grand-challenge.org",
        image_format="MetaImage .mha full CTs (~221 GB) or 64x128x128 nodule blocks (~6.8 GB)",
        annotation_file="LUNA25_Public_Training_Development_Data.csv",
        columns={
            "uid": "SeriesInstanceUID",
            "x": "CoordX",
            "y": "CoordY",
            "z": "CoordZ",
            "label": "label",
            "patient": "PatientID",
            "nodule": "NoduleID",
            "date": "StudyDate",
        },
        sources=("https://zenodo.org/records/14223624", "https://zenodo.org/records/14673658"),
        size="2,120 patients, 4,069 scans, 3,762 nodules, 6,163 annotations (9.0% malignant)",
        pitfalls=(
            "Labels are effectively patient-level: every nodule of a cancer patient is malignant.",
            "Annotation count per patient alone predicts cancer (AUC ~0.76). Always split by PatientID.",
            "The same nodule appears in up to 3 screening rounds (StudyDate 1999/2000/2001); "
            "report per-round and one-row-per-nodule metrics, not only pooled rows.",
            "Point annotations only: no masks, diameters or radiologist descriptors.",
            "Slice thickness varies (1.25 / 2.0 / 2.5 mm): resample before training.",
            "Some Kaggle mirrors mislabel the licence as CC0 or ship an empty annotation CSV.",
        ),
        citation="Peeters et al., LUNA25 challenge, MICCAI 2025. Images from NLST (NEJM 2011;365:395-409).",
    ),
    "lidc-idri": DatasetInfo(
        name="lidc-idri",
        full_name="Lung Image Database Consortium and Image Database Resource Initiative",
        task="Nodule detection, segmentation and characterisation (4-reader ratings)",
        licence="CC BY 3.0",
        homepage="https://www.cancerimagingarchive.net/collection/lidc-idri/",
        image_format="DICOM series with per-reader XML annotations",
        annotation_file="XML files (one per scan), easiest to read with the pylidc package",
        columns={},
        size="1,018 CT scans from 1,010 patients",
        loader=False,
        pitfalls=(
            "Up to 4 readers mark each nodule separately; cluster marks before counting nodules.",
            "Malignancy scores (1-5) are reader opinions, not pathology-confirmed diagnoses.",
        ),
        citation="Armato et al., Medical Physics 38(2) (2011) 915-931.",
    ),
}


def get_info(name: str) -> DatasetInfo:
    key = name.lower()
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    return DATASETS[key]
