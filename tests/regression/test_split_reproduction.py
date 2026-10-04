"""Thuật toán chia phải giữ ĐÚNG như bản tham chiếu (chép nguyên văn bên dưới): GAN đã train (ISIC checkpoints_v5;
RSNA checkpoints_rsna, gắn với split rsna_v2 qua `split.expected`) chỉ được thấy phần train của đúng split đó —
đổi thuật toán / thứ tự gọi RNG = split khác = GAN có thể đã thấy ảnh val/test."""

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


def _reference_stratified(pp, classes, test_fraction, val_fraction, seed, group_regex=None):
    """create_split tham chiếu (từ notebook gốc), chép nguyên văn — khoá thuật toán của `stratified`."""
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


def test_stratified_matches_reference(gray_tree):
    """Đúng tham số của RSNA (test 0,15, val 0,176, seed 2026) — split rsna_v2 mà GAN rsna đã dùng."""
    classes = ["negative", "positive"]
    assert stratified(gray_tree, classes, 0.176, 0.15, 2026) == \
        _reference_stratified(gray_tree, classes, 0.15, 0.176, 2026)


def test_grouped_split_matches_reference_and_keeps_patients_together(tmp_path):
    root = tmp_path / "pp"
    for label, n in [("benign", 8), ("malignant", 5)]:
        (root / label).mkdir(parents=True)
        for p in range(n):
            for s in range(3):
                (root / label / f"P{label[0]}{p:02d}_slice{s}.png").write_bytes(b"x")
    regex = r"^(P\w\d+)_"
    split = stratified(root, CLASSES, 0.25, 0.2, 3, regex)
    assert split == _reference_stratified(root, CLASSES, 0.2, 0.25, 3, regex)
    for parts in split.values():
        patients = {s: {f.split("_")[0] for f in files} for s, files in parts.items()}
        assert not (patients["train"] & patients["test"]) and not (patients["train"] & patients["val"])
