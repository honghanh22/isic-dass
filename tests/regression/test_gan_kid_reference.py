"""KID dừng sớm GAN phải so với ảnh thật lớp thiểu số của TRAIN, không dùng val.

Val chỉ để chọn epoch classifier; dùng val để chọn snapshot GAN thì val tham gia hai lựa chọn mô hình (và lớp thiểu số
của val nhỏ hơn -> KID nhiễu hơn). Stage `gan` cần GPU nên kiểm tra trên mã nguồn.
"""

from __future__ import annotations

import ast
import inspect

from dass.pipeline.stages import gan


def test_gan_kid_reference_uses_train_only():
    src = inspect.getsource(gan.run)
    assert "val_paths" not in src and "val_pp" not in src, "stage gan không được đọc ảnh val"
    assigns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "real_feats" for t in n.targets)]
    assert assigns and "train_paths" in ast.unparse(assigns[0].value)
