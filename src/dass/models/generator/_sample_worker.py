"""Tiến trình con sinh ảnh (tách bộ nhớ GPU của PyTorch khỏi tiến trình chính).

    python _sample_worker.py <repo> <pkl> <out_dir> <class_idx> <n_images> <psi> <seed> <batch> <channels>
                             <tolerance> <force_gray>

Ảnh lưu đúng `channels` của bộ dữ liệu. GAN 3 kênh + dữ liệu 1 kênh: chỉ gộp khi 3 kênh giống nhau
(chênh lệch ≤ tolerance mức xám); ngược lại thoát mã 3 (không được tự ý "đổi màu" ảnh).
`force_gray` = 1: chuyển về luminance (giống hệt ảnh thật đã tiền xử lý) rồi lưu `channels` kênh bằng nhau.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

EXIT_CHANNEL_MISMATCH = 3


def main(argv: list[str]) -> None:
    repo, pkl, out_dir, class_idx, n_images, psi, seed, batch, channels, tolerance, force_gray = argv
    class_idx, n_images, seed, batch = int(class_idx), int(n_images), int(seed), int(batch)
    channels, tolerance, psi, force_gray = int(channels), int(tolerance), float(psi), force_gray == "1"
    sys.path.insert(0, repo)
    import legacy

    device = torch.device("cuda")
    with open(pkl, "rb") as fh:
        G = legacy.load_network_pkl(fh)["G_ema"].to(device).eval().requires_grad_(False)
    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(seed)
    done, worst = 0, 0
    with torch.no_grad():
        while done < n_images:
            b = min(batch, n_images - done)
            z = torch.randn([b, G.z_dim], generator=gen, device=device)
            c = torch.zeros([b, G.c_dim], device=device)
            c[:, class_idx] = 1
            img = G(z, c, truncation_psi=psi, noise_mode="const")
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            if img.shape[-1] == 3:
                worst = max(worst, int(np.abs(np.diff(img.astype(np.int16), axis=-1)).max()))
            if force_gray and img.shape[-1] == 3:     # cùng phép chuyển luminance của PIL như ảnh thật
                img = np.stack([np.asarray(Image.fromarray(a).convert("L")) for a in img])[..., None]
                if channels == 3:
                    img = np.repeat(img, 3, axis=-1)
            elif channels == 1 and img.shape[-1] == 3:
                diff = int(np.abs(np.diff(img.astype(np.int16), axis=-1)).max())
                if diff > tolerance:
                    print(f"CHANNEL_MISMATCH max_diff={diff} tolerance={tolerance} at image {done}", flush=True)
                    sys.exit(EXIT_CHANNEL_MISMATCH)
                img = img[..., :1]
            elif channels == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            for j in range(b):
                arr = img[j]
                pil = Image.fromarray(np.ascontiguousarray(arr[..., 0] if arr.shape[-1] == 1 else arr))
                pil.save(os.path.join(out_dir, f"synth_{done + j:05d}.png"), format="PNG")
            done += b
            if done % 200 == 0 or done == n_images:
                print(f"{done}/{n_images}", flush=True)
    print(f"DONE {done} generator_channels={G.img_channels} saved_channels={channels} force_gray={int(force_gray)} "
          f"max_channel_diff_before={worst}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
