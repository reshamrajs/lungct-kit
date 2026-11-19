"""Synthetic LUNA-style data: small CT volumes with a bright sphere ("nodule") at a known world point."""
import csv

import numpy as np
import pytest

from lungct_kit import Volume, voxel_to_world, write_volume

LUNA25_HEADER = ["PatientID", "SeriesInstanceUID", "StudyDate", "CoordX", "CoordY", "CoordZ",
                 "LesionID", "AnnotationID", "NoduleID", "label", "Age_at_StudyDate", "Gender"]


def make_scan(center_zyx=(10, 30, 30), shape=(24, 60, 60), spacing=(2.5, 0.7, 0.7),
              origin=(-300.0, -120.0, -150.0), radius_mm=4.0, direction=None):
    arr = np.full(shape, -800, dtype=np.int16)  # lung parenchyma
    zz, yy, xx = np.indices(shape)
    dist_mm = np.sqrt(((zz - center_zyx[0]) * spacing[0]) ** 2 + ((yy - center_zyx[1]) * spacing[1]) ** 2
                      + ((xx - center_zyx[2]) * spacing[2]) ** 2)
    arr[dist_mm <= radius_mm] = 50  # soft-tissue nodule
    vol = Volume(arr, spacing, origin, np.eye(3) if direction is None else direction)
    return vol, voxel_to_world(center_zyx, vol)


@pytest.fixture
def luna25_dir(tmp_path):
    """6 patients; patients p0 and p1 are 'cancer' (all their rows label 1). Some have 2 rounds."""
    img_dir = tmp_path / "images"
    rows = []
    plan = {"p0": 2, "p1": 1, "p2": 2, "p3": 1, "p4": 1, "p5": 2}
    for p_i, (pid, rounds) in enumerate(plan.items()):
        for r in range(rounds):
            uid = f"1.2.840.{p_i}.{r}"
            center = (8 + r, 25 + p_i, 28)
            vol, world = make_scan(center_zyx=center)
            write_volume(vol, img_dir / f"{uid}.mha")
            date = 19990102 + r * 10000
            rows.append({"PatientID": pid, "SeriesInstanceUID": uid, "StudyDate": date,
                         "CoordX": world[0], "CoordY": world[1], "CoordZ": world[2], "LesionID": 1,
                         "AnnotationID": f"{pid}_1_{date}", "NoduleID": f"{pid}_1",
                         "label": 1 if pid in ("p0", "p1") else 0, "Age_at_StudyDate": 60, "Gender": "Male"})
    csv_path = tmp_path / "LUNA25_Public_Training_Development_Data.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, LUNA25_HEADER)
        w.writeheader()
        w.writerows(rows)
    return csv_path, img_dir


@pytest.fixture
def luna16_dir(tmp_path):
    img_dir = tmp_path / "luna16"
    rows = []
    for s in range(3):
        uid = f"1.3.6.1.{s}"
        vol, world = make_scan(center_zyx=(12, 30, 20 + s))
        write_volume(vol, img_dir / f"subset{s}" / f"{uid}.mhd")
        rows.append([uid, *world, 8.0])
    csv_path = tmp_path / "annotations.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
        w.writerows(rows)
    return csv_path, img_dir
