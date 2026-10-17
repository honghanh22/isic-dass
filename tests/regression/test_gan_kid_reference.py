"""Chỉ số chọn snapshot / dừng sớm GAN phải so với ảnh thật của TRAIN, không dùng val.

Val chỉ để chọn epoch classifier; dùng val để chọn snapshot GAN thì val tham gia hai lựa chọn mô hình (và val nhỏ hơn
-> điểm nhiễu hơn). Áp dụng cho mọi `generator.select_metric` (kid_inception, fid_ed). Stage `gan` cần GPU nên kiểm
tra trên mã nguồn.
"""

from __future__ import annotations

import ast
import inspect

from dass.pipeline.stages import gan


def test_gan_selection_reference_uses_train_only():
    for fn in (gan.run, gan.build_scorer):
        src = inspect.getsource(fn)
        assert "val_paths" not in src and "val_pp" not in src, f"{fn.__name__} không được đọc ảnh val"
    src = inspect.getsource(gan.build_scorer)
    assigns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "real" for t in n.targets)]
    assert len(assigns) == 2, "mỗi select_metric gán đặc trưng ảnh thật đúng một lần"
    assert all("ctx.train_paths" in ast.unparse(a.value) for a in assigns)
