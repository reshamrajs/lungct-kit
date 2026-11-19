"""Command-line tool: ``lungct list | info | check | split | export``."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import __version__
from .datasets import NoduleDataset, index_images, luna16_subset, read_annotations
from .registry import DATASETS, get_info
from .splits import check_no_leakage, group_kfold


def cmd_list(_: argparse.Namespace) -> int:
    for d in DATASETS.values():
        status = "loader" if d.loader else "info only"
        print(f"{d.name:10} {d.full_name}  [{status}]")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    d = get_info(args.dataset)
    print(f"{d.full_name} ({d.name})\n")
    for label, value in [("Task", d.task), ("Size", d.size), ("Licence", d.licence),
                         ("Images", d.image_format), ("Annotations", d.annotation_file),
                         ("Homepage", d.homepage), ("Cite", d.citation)]:
        if value:
            print(f"  {label:12} {value}")
    for s in d.sources:
        print(f"  {'Download':12} {s}")
    if d.pitfalls:
        print("\n  Pitfalls:")
        for p in d.pitfalls:
            print(f"   - {p}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    nodules = read_annotations(args.annotations, args.dataset)
    images = index_images(args.images)
    have = [n for n in nodules if n.uid in images]
    scans = {n.uid for n in nodules}
    print(f"annotations : {len(nodules)} rows, {len(scans)} scans, {len({n.group for n in nodules})} groups")
    present = len(scans & set(images))
    print(f"images found: {len(images)} files; {present}/{len(scans)} annotated scans present")
    labels = Counter(n.label for n in have)
    if any(k is not None for k in labels):
        print("labels      : " + ", ".join(f"{k}: {v}" for k, v in sorted(labels.items(), key=str)))
    if args.dataset == "luna16":
        folds = Counter(luna16_subset(images[n.uid]) for n in have)
        print("subsets     : " + ", ".join(f"subset{k}: {v}" for k, v in sorted(folds.items(), key=str)))
    missing = len(nodules) - len(have)
    if missing:
        print(f"WARNING: {missing} annotations point to scans that were not found under {args.images}")
    return 0 if not missing or args.allow_missing else 1


def cmd_split(args: argparse.Namespace) -> int:
    nodules = read_annotations(args.annotations, args.dataset)
    groups = [n.group for n in nodules]
    labels = [n.label or 0 for n in nodules] if any(n.label is not None for n in nodules) else None
    folds = group_kfold(groups, labels, n_folds=args.folds, seed=args.seed)
    out = {"dataset": args.dataset, "grouped_by": "patient" if nodules[0].patient else "scan",
           "seed": args.seed, "folds": []}
    for k, (train, test) in enumerate(folds):
        check_no_leakage(groups, train, test)
        test_groups = sorted({groups[i] for i in test})
        pos = sum(labels[i] for i in test) if labels else None
        out["folds"].append({"fold": k, "test_groups": test_groups, "n_test_rows": len(test),
                             "n_test_positive": pos})
        extra = f", {pos} positive" if pos is not None else ""
        print(f"fold {k}: {len(test_groups)} groups, {len(test)} rows{extra}")
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} (no group appears in more than one test fold)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    ds = NoduleDataset(read_annotations(args.annotations, args.dataset), index_images(args.images),
                       patch_size=args.patch, spacing=args.spacing, window=None)
    if not len(ds):
        print("no nodules matched any image", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    patches = np.stack([ds[i]["image"][0].astype(np.int16) for i in range(len(ds))])
    np.savez_compressed(
        out, patches=patches,
        labels=np.array([-1 if n.label is None else n.label for n in ds.nodules]),
        uids=np.array([n.uid for n in ds.nodules]), groups=np.array(ds.groups),
        spacing=np.array(args.spacing),
    )
    print(f"wrote {len(ds)} patches of {patches.shape[1:]} HU voxels to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lungct", description="Tools for public lung CT nodule datasets.")
    p.add_argument("--version", action="version", version=f"lungct-kit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list supported datasets").set_defaults(func=cmd_list)

    s = sub.add_parser("info", help="licence, size, download links and pitfalls of a dataset")
    s.add_argument("dataset", choices=list(DATASETS))
    s.set_defaults(func=cmd_info)

    loadable = [k for k, d in DATASETS.items() if d.loader]
    for name, func, help_ in [("check", cmd_check, "check annotations against downloaded images"),
                              ("split", cmd_split, "write leakage-safe patient-grouped folds"),
                              ("export", cmd_export, "export nodule patches to a .npz file")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("dataset", choices=loadable)
        s.add_argument("--annotations", required=True, help="annotation CSV")
        if name != "split":
            s.add_argument("--images", required=True, help="folder containing the scans")
        s.set_defaults(func=func)
        if name == "check":
            s.add_argument("--allow-missing", action="store_true")
        if name == "split":
            s.add_argument("--folds", type=int, default=5)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--out", default="splits.json")
        if name == "export":
            s.add_argument("--patch", type=int, default=64, help="patch edge in voxels")
            s.add_argument("--spacing", type=float, default=1.0, help="isotropic spacing in mm")
            s.add_argument("--out", default="patches.npz")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
