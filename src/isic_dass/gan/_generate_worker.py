"""Tiến trình con sinh ảnh (tách bộ nhớ GPU của PyTorch khỏi tiến trình chính).

Cách gọi: python _generate_worker.py <repo> <pkl> <out_dir> <class_idx> <n_images> <psi> <seed> <batch>
"""

import os
import sys

import torch
from PIL import Image


def main(argv: list[str]) -> None:
    repo, pkl, out_dir, class_idx, n_images, psi, seed, batch = argv
    class_idx, n_images, seed, batch = int(class_idx), int(n_images), int(seed), int(batch)
    psi = float(psi)
    sys.path.insert(0, repo)
    import legacy

    device = torch.device("cuda")
    with open(pkl, "rb") as fh:
        G = legacy.load_network_pkl(fh)["G_ema"].to(device).eval().requires_grad_(False)
    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(seed)
    done = 0
    with torch.no_grad():
        while done < n_images:
            b = min(batch, n_images - done)
            z = torch.randn([b, G.z_dim], generator=gen, device=device)
            c = torch.zeros([b, G.c_dim], device=device)
            c[:, class_idx] = 1
            img = G(z, c, truncation_psi=psi, noise_mode="const")
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            for j in range(b):
                Image.fromarray(img[j]).save(os.path.join(out_dir, f"synth_{done + j:05d}.png"), format="PNG")
            done += b
            if done % 200 == 0 or done == n_images:
                print(f"{done}/{n_images}", flush=True)
    print(f"DONE {done}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
