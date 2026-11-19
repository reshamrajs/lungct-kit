import json

import numpy as np
import pytest

from lungct_kit import (
    LeakageError,
    check_no_leakage,
    group_kfold,
    index_images,
    load,
    luna16_subset,
    read_annotations,
)
from lungct_kit.cli import main

# ---------- splits ----------

def test_group_kfold_never_leaks_and_covers_everything():
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 40, size=300).tolist()
    labels = [int(g < 8) for g in groups]  # patient-level labels like LUNA25
    folds = group_kfold(groups, labels, n_folds=5, seed=3)
    tests = np.concatenate([t for _, t in folds])
    assert sorted(tests.tolist()) == list(range(300))
    for train, test in folds:
        check_no_leakage(groups, train, test)
        assert {groups[i] for i in test if labels[i]}  # every fold gets positive patients


def test_group_kfold_stratifies_positive_patients():
    groups = [f"p{i}" for i in range(50)]
    labels = [1] * 10 + [0] * 40
    for _, test in group_kfold(groups, labels, n_folds=5):
        assert sum(labels[i] for i in test) == 2


def test_group_kfold_is_deterministic_and_validates():
    groups = list("aabbccddeeff")
    first, second = group_kfold(groups, seed=7), group_kfold(groups, seed=7)
    assert all((a[1] == b[1]).all() for a, b in zip(first, second, strict=True))
    with pytest.raises(ValueError):
        group_kfold(groups, n_folds=7)
    with pytest.raises(ValueError):
        group_kfold(groups, labels=[0, 1])


def test_check_no_leakage_raises():
    with pytest.raises(LeakageError):
        check_no_leakage(["a", "a", "b"], [0], [1, 2])


# ---------- annotations and datasets ----------

def test_read_luna25(luna25_dir):
    csv_path, _ = luna25_dir
    nodules = read_annotations(csv_path, "luna25")
    assert len(nodules) == 9
    assert {n.group for n in nodules} == {f"p{i}" for i in range(6)}
    assert nodules[0].extra["NoduleID"] == "p0_1"
    assert sum(n.label for n in nodules) == 3


def test_read_rejects_wrong_columns(luna16_dir):
    csv_path, _ = luna16_dir
    with pytest.raises(ValueError, match="missing columns"):
        read_annotations(csv_path, "luna25")
    with pytest.raises(ValueError):
        read_annotations(csv_path, "lidc-idri")


def test_luna25_patches_are_centred_on_the_nodule(luna25_dir):
    csv_path, img_dir = luna25_dir
    ds = load("luna25", csv_path, img_dir, patch_size=(16, 32, 32), spacing=1.0)
    assert len(ds) == 9 and not ds.missing
    item = ds[0]
    assert item["image"].shape == (1, 16, 32, 32)
    assert item["image"].dtype == np.float32
    assert 0.0 <= item["image"].min() and item["image"].max() <= 1.0
    # nodule (50 HU -> 0.75) at the centre, lung (-800 HU -> ~0.14) at the corner
    assert item["image"][0, 8, 16, 16] == pytest.approx((50 + 1000) / 1400, abs=0.02)
    assert item["image"][0, 0, 0, 0] < 0.2

    folds = group_kfold(ds.groups, ds.labels, n_folds=3)
    train, test = folds[0]
    check_no_leakage(ds.groups, train, test)
    assert len(ds.subset(test)) == len(test)


def test_luna16_index_and_subsets(luna16_dir):
    csv_path, img_dir = luna16_dir
    images = index_images(img_dir)
    assert len(images) == 3
    assert sorted(luna16_subset(p) for p in images.values()) == [0, 1, 2]
    ds = load("luna16", csv_path, img_dir, patch_size=16)
    assert ds[2]["label"] is None and ds[2]["image"][0, 8, 8, 8] > 0.7


def test_missing_images_are_reported(luna25_dir):
    csv_path, img_dir = luna25_dir
    next(img_dir.glob("*.mha")).unlink()
    ds = load("luna25", csv_path, img_dir)
    assert len(ds.missing) == 1 and len(ds) == 8
    with pytest.raises(FileNotFoundError):
        load("luna25", csv_path, img_dir, drop_missing=False)


# ---------- CLI ----------

def test_cli_list_and_info(capsys):
    assert main(["list"]) == 0
    assert "luna25" in capsys.readouterr().out
    assert main(["info", "luna25"]) == 0
    out = capsys.readouterr().out
    assert "PatientID" in out and "zenodo.org" in out


def test_cli_check_split_export(luna25_dir, tmp_path, capsys):
    csv_path, img_dir = luna25_dir
    assert main(["check", "luna25", "--annotations", str(csv_path), "--images", str(img_dir)]) == 0
    assert "9/9 annotated scans present" in capsys.readouterr().out
    out_json = tmp_path / "splits.json"
    split_args = ["split", "luna25", "--annotations", str(csv_path), "--folds", "3", "--out", str(out_json)]
    assert main(split_args) == 0
    splits = json.loads(out_json.read_text())
    assert splits["grouped_by"] == "patient"
    all_groups = [g for f in splits["folds"] for g in f["test_groups"]]
    assert len(all_groups) == len(set(all_groups)) == 6

    out_npz = tmp_path / "patches.npz"
    assert main(["export", "luna25", "--annotations", str(csv_path), "--images", str(img_dir),
                 "--patch", "16", "--out", str(out_npz)]) == 0
    data = np.load(out_npz)
    assert data["patches"].shape == (9, 16, 16, 16) and data["patches"].dtype == np.int16
    assert data["labels"].sum() == 3
