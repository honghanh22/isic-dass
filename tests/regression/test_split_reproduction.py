"""Split phải tái lập ĐÚNG thuật toán của hai notebook gốc: GAN cũ (checkpoints_v5, checkpoints_bt) đã được train
trên phần train của các split đó — lệch split = GAN có thể đã thấy ảnh val/test."""

import os
import re

import numpy as np

from dass.data.splits import holdout_val, stratified

CLASSES = ["benign", "malignant"]


def _isic_v5(pp, classes, val_fraction, seed):
    """create_real_val_split — notebook ISIC v5 (cell 11), chép nguyên văn."""
    rng = np.random.default_rng(seed)
    split = {}
    for label in classes:
        files = sorted(os.listdir(pp / label))
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_fraction))
        split[label] = {"val": files[:n_val], "train": files[n_val:]}
    return split


def _brain_v1(pp, classes, test_fraction, val_fraction, seed, group_regex=None):
    """create_split — notebook Brain Tumor v1 (cell 9), chép nguyên văn."""
    rng = np.random.default_rng(seed)
    split = {}
    for label in classes:
        files = sorted(os.listdir(pp / label))
        if group_regex:
            groups = {}
            for f in files:
                m = re.match(group_regex, f)
                groups.setdefault(m.group(1) if m else f, []).append(f)
            keys = sorted(groups)
            rng.shuffle(keys)
            n_test_g = max(1, int(len(keys) * test_fraction))
            n_val_g = max(1, int((len(keys) - n_test_g) * val_fraction))
            take = lambda ks: [f for k in ks for f in groups[k]]  # noqa: E731, B023 — chép nguyên văn notebook
            split[label] = {"test": take(keys[:n_test_g]), "val": take(keys[n_test_g:n_test_g + n_val_g]),
                            "train": take(keys[n_test_g + n_val_g:])}
        else:
            files = list(files)
            rng.shuffle(files)
            n_test = max(1, int(len(files) * test_fraction))
            n_val = max(1, int((len(files) - n_test) * val_fraction))
            split[label] = {"test": files[:n_test], "val": files[n_test:n_test + n_val],
                            "train": files[n_test + n_val:]}
    return split


def test_holdout_val_reproduces_isic_v5(image_tree):
    assert holdout_val(image_tree, CLASSES, 0.15, 2026) == _isic_v5(image_tree, CLASSES, 0.15, 2026)


def test_stratified_reproduces_brain_tumor_v1(gray_tree):
    classes = ["negative", "positive"]
    assert stratified(gray_tree, classes, 0.176, 0.15, 2026) == _brain_v1(gray_tree, classes, 0.15, 0.176, 2026)


def test_grouped_split_reproduces_v1_and_keeps_patients_together(tmp_path):
    root = tmp_path / "pp"
    for label, n in [("benign", 8), ("malignant", 5)]:
        (root / label).mkdir(parents=True)
        for p in range(n):
            for s in range(3):
                (root / label / f"P{label[0]}{p:02d}_slice{s}.png").write_bytes(b"x")
    regex = r"^(P\w\d+)_"
    split = stratified(root, CLASSES, 0.25, 0.2, 3, regex)
    assert split == _brain_v1(root, CLASSES, 0.2, 0.25, 3, regex)
    for parts in split.values():
        patients = {s: {f.split("_")[0] for f in files} for s, files in parts.items()}
        assert not (patients["train"] & patients["test"]) and not (patients["train"] & patients["val"])
