"""Leakage-safe train/test splits.

Lung nodule datasets have several nodules (and, in LUNA25, several screening
rounds) per patient. Splitting by row or by nodule puts the same patient in
train and test and inflates AUC. In LUNA25 the labels are effectively
patient-level, so the number of rows per patient alone gives AUC ~0.76.
Always split by patient.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

import numpy as np


class LeakageError(ValueError):
    """Raised when the same group (patient) appears in more than one split."""


def group_kfold(
    groups: Sequence[Hashable],
    labels: Sequence[int] | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``n_folds`` ``(train_idx, test_idx)`` pairs with no group in both.

    If ``labels`` are given, folds are stratified at the group level: a group is
    "positive" if any of its rows is positive, and positive and negative groups
    are dealt out separately so every fold gets a similar share. Within each
    class, groups are assigned largest-first to the fold with the fewest rows,
    which keeps fold sizes balanced. Deterministic for a given ``seed``.
    """
    groups = list(groups)
    n = len(groups)
    if labels is not None and len(labels) != n:
        raise ValueError("labels and groups must have the same length")
    rows_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        rows_by_group[g].append(i)
    if n_folds < 2 or n_folds > len(rows_by_group):
        raise ValueError(f"n_folds must be between 2 and the number of groups ({len(rows_by_group)})")

    rng = np.random.default_rng(seed)
    keys = list(rows_by_group)
    rng.shuffle(keys)  # random tie-break between equal-sized groups

    if labels is None:
        strata = [keys]
    else:
        pos = [g for g in keys if any(labels[i] for i in rows_by_group[g])]
        pos_set = set(pos)
        strata = [pos, [g for g in keys if g not in pos_set]]

    fold_rows = [0] * n_folds
    assignment: dict[Hashable, int] = {}
    for stratum in strata:
        fold_groups = [0] * n_folds
        for g in sorted(stratum, key=lambda g: -len(rows_by_group[g])):
            f = min(range(n_folds), key=lambda k: (fold_groups[k], fold_rows[k]))
            assignment[g] = f
            fold_groups[f] += 1
            fold_rows[f] += len(rows_by_group[g])

    fold_of_row = np.array([assignment[g] for g in groups])
    all_idx = np.arange(n)
    return [(all_idx[fold_of_row != k], all_idx[fold_of_row == k]) for k in range(n_folds)]


def check_no_leakage(groups: Sequence[Hashable], *index_sets: Sequence[int]) -> None:
    """Raise :class:`LeakageError` if any group appears in more than one index set."""
    seen: dict[Hashable, int] = {}
    for set_no, idx in enumerate(index_sets):
        for g in {groups[i] for i in idx}:
            if g in seen and seen[g] != set_no:
                raise LeakageError(f"group {g!r} is in split {seen[g]} and split {set_no}")
            seen[g] = set_no
