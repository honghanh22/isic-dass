"""Train StyleGAN2-ADA có điều kiện với early stopping theo KID của lớp thiểu số.

- `train.py` chính thức chạy trong tiến trình con; cứ `kimg_per_snap` kimg lưu một snapshot.
- Mỗi snapshot được đánh giá KID; snapshot tốt nhất copy lên Drive (`best.pkl`), mới nhất lưu `latest.pkl`.
- Sau `min_kimg`, nếu KID không giảm ≥ `min_rel_delta` trong `patience` snapshot liên tiếp -> dừng.
- Colab ngắt: gọi lại `train()` -> resume từ `latest.pkl`, giữ lịch sử KID và bộ đếm patience.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ...config import Layout
from ...config.schema import GeneratorConfig
from ...evaluation.generative import kid_from_features
from ...utils import read_json, write_json_atomic
from .inception import generate_class_images, inception_features, load_generator

log = logging.getLogger(__name__)


def default_state() -> dict:
    return {"cum_kimg": 0, "history": [], "best_kid": None, "best_cum_kimg": None,
            "bad_count": 0, "finished": False, "stop_reason": None}


def load_state(path: str | Path) -> dict:
    return read_json(path, default=None) or default_state()


def _list_run_dirs(outdir: Path) -> list[Path]:
    return sorted(d for d in outdir.glob("0*") if d.is_dir())


def _list_snapshots(run_dir: Path) -> list[tuple[int, Path]]:
    return sorted((int(p.stem.split("-")[-1]), p) for p in run_dir.glob("network-snapshot-*.pkl"))


def _is_file_complete(path: Path, wait: int = 10) -> bool:
    """File snapshot đang được ghi nếu kích thước còn thay đổi sau `wait` giây."""
    try:
        s1 = path.stat().st_size
        time.sleep(wait)
        s2 = path.stat().st_size
    except OSError:
        return False
    return s1 > 0 and s1 == s2


def _last_tick_line(log_path: Path) -> str:
    try:
        ticks = [line.strip() for line in log_path.read_text().splitlines() if line.startswith("tick")]
        return ticks[-1] if ticks else "(chưa có tick)"
    except OSError:
        return "(chưa có log)"


class StyleGanTrainer:
    def __init__(self, layout: Layout, gan: GeneratorConfig, seed: int, real_features: np.ndarray, eval_class_idx: int):
        self.layout = layout
        self.gan = gan
        self.seed = seed
        self.real_features = real_features      # Inception features của ảnh thật lớp thiểu số (train)
        self.eval_class_idx = eval_class_idx

    # ------------------------------------------------------------------ public
    def train(self, fresh_start: bool = False) -> dict:
        if fresh_start:
            self._reset()
        self.layout.gan_runs.mkdir(parents=True, exist_ok=True)
        self.layout.gan_dir.mkdir(parents=True, exist_ok=True)

        state = load_state(self.layout.gan_state_json)
        if state["finished"]:
            log.info("GAN đã train xong (%s). Best KID = %.5f tại %s kimg.",
                     state["stop_reason"], state["best_kid"], state["best_cum_kimg"])
            return state

        while not state["finished"]:
            offset = state["cum_kimg"]
            remaining = self.gan.max_kimg - offset
            if remaining < self.gan.kimg_per_snap:
                state.update(finished=True, stop_reason=f"đạt max_kimg={self.gan.max_kimg}")
                self._save(state)
                break

            resume = None
            if offset > 0:
                if not self.layout.gan_latest_pkl.exists():
                    raise RuntimeError("Có lịch sử train nhưng thiếu latest.pkl trên Drive -> chạy lại với fresh_start.")
                resume = self.layout.gan_latest_pkl
                log.info("Resume từ %s (đã train %d kimg)", resume, offset)

            early_stopped, ret, log_path = self._run_segment(self._command(remaining, resume), offset, state)
            if early_stopped:
                log.info("DỪNG SỚM: %s", state["stop_reason"])
                break
            if ret != 0:
                print("\n".join(log_path.read_text().splitlines()[-60:]))
                raise RuntimeError(f"train.py lỗi (mã {ret}). Xem log: {log_path}")
            if state["cum_kimg"] == offset:
                raise RuntimeError(f"train.py kết thúc nhưng không có snapshot mới. Xem log: {log_path}")

        log.info("Kết thúc. Best KID = %.5f tại %s kimg -> %s",
                 state["best_kid"], state["best_cum_kimg"], self.layout.gan_best_pkl)
        return state

    def evaluate_snapshot(self, pkl_path: Path) -> float:
        import torch

        G = load_generator(pkl_path)
        fake = generate_class_images(G, self.gan.kid_n_gen, self.eval_class_idx, psi=1.0, seed=self.gan.kid_seed)
        del G
        torch.cuda.empty_cache()
        return kid_from_features(self.real_features, inception_features(fake),
                                 num_subsets=self.gan.kid_subsets, seed=self.gan.kid_seed)

    # ----------------------------------------------------------------- private
    def _save(self, state: dict) -> None:
        write_json_atomic(self.layout.gan_state_json, state)

    def _reset(self) -> None:
        for p in [self.layout.gan_state_json, self.layout.gan_best_pkl, self.layout.gan_latest_pkl]:
            p.unlink(missing_ok=True)
        shutil.rmtree(self.layout.gan_runs, ignore_errors=True)
        log.warning("fresh_start -> đã xoá trạng thái GAN cũ, train từ đầu.")

    def _command(self, remaining: int, resume: Path | None) -> list[str]:
        g = self.gan
        snap_ticks = max(1, g.kimg_per_snap // 4)   # 1 tick = 4 kimg (mặc định của tác giả)
        cmd = [sys.executable, "train.py",
               f"--outdir={self.layout.gan_runs}", f"--data={self.layout.gan_dataset_zip}",
               "--gpus=1", "--cond=1", f"--mirror={int(g.mirror)}",
               f"--cfg={g.cfg}", f"--batch={g.batch}", f"--gamma={g.gamma}",
               f"--kimg={remaining}", f"--snap={snap_ticks}", "--metrics=none",
               "--aug=ada", f"--target={g.aug_target}", f"--seed={self.seed}", f"--workers={g.workers}"]
        if resume:
            cmd.append(f"--resume={resume}")
        return cmd

    def _run_segment(self, cmd: list[str], offset: int, state: dict) -> tuple[bool, int | None, Path]:
        runs = self.layout.gan_runs
        existing = set(_list_run_dirs(runs))
        log_path = runs / f"train_log_from_{offset}kimg.txt"
        log.info("Lệnh: %s", " ".join(cmd))

        run_dir: Path | None = None
        evaluated: set[Path] = set()
        early_stopped = False
        ret = None
        last_print = time.time()
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(cmd, cwd=self.layout.sg2_repo, stdout=log_f, stderr=subprocess.STDOUT)
            try:
                while True:
                    ret = proc.poll()
                    if run_dir is None:
                        new_runs = [d for d in _list_run_dirs(runs) if d not in existing]
                        if new_runs:
                            run_dir = new_runs[0]
                            log.info("Thư mục run: %s", run_dir)
                    if run_dir is not None:
                        early_stopped = self._consume_snapshots(run_dir, offset, evaluated, state, log_path)
                    if early_stopped or ret is not None:
                        break
                    if time.time() - last_print > 600:
                        log.info("  ... %s", _last_tick_line(log_path))
                        last_print = time.time()
                    time.sleep(self.gan.poll_seconds)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        proc.kill()

        if not early_stopped and run_dir is not None:     # snapshot cuối (lưu khi train.py kết thúc)
            early_stopped = self._consume_snapshots(run_dir, offset, evaluated, state, log_path)
        self._backup_logs(run_dir, log_path)
        return early_stopped, ret, log_path

    def _backup_logs(self, run_dir: Path | None, log_path: Path) -> None:
        """Sao lưu log train + lưới ảnh mẫu (fakes*.png) và stats*.jsonl của StyleGAN2-ADA lên Drive."""
        dst = self.layout.gan_logs_dir
        try:
            dst.mkdir(parents=True, exist_ok=True)
            if log_path.exists():
                shutil.copy2(log_path, dst / log_path.name)
            if run_dir is not None:
                run_id = run_dir.name.split("-")[0]
                for p in [*run_dir.glob("*.png"), *run_dir.glob("*.jsonl")]:
                    shutil.copy2(p, dst / f"run{run_id}_{p.name}")
        except OSError as e:
            log.warning("Không sao lưu được log / ảnh mẫu GAN: %s", e)

    def _consume_snapshots(self, run_dir: Path, offset: int, evaluated: set[Path], state: dict,
                           log_path: Path) -> bool:
        """Đánh giá các snapshot mới. Trả về True nếu kích hoạt early stopping."""
        g = self.gan
        for kimg, pkl in _list_snapshots(run_dir):
            if pkl in evaluated:
                continue
            if kimg == 0:                          # snapshot khởi tạo / mạng vừa resume -> bỏ qua
                evaluated.add(pkl)
                continue
            if not _is_file_complete(pkl):
                return False
            evaluated.add(pkl)
            cum = offset + kimg
            kid = self.evaluate_snapshot(pkl)
            improved = state["best_kid"] is None or kid < state["best_kid"] * (1 - g.min_rel_delta)
            if improved:
                shutil.copy2(pkl, self.layout.gan_best_pkl)
                state.update(best_kid=kid, best_cum_kimg=cum, bad_count=0)
            elif cum >= g.min_kimg:
                state["bad_count"] += 1
            shutil.copy2(pkl, self.layout.gan_latest_pkl)
            state["cum_kimg"] = cum
            state["history"].append({"cum_kimg": cum, "kid": kid, "improved": bool(improved)})
            self._save(state)
            pkl.unlink()                           # đã lưu best/latest trên Drive, xoá bản cục bộ cho nhẹ đĩa
            self._backup_logs(run_dir, log_path)
            log.info("[snapshot] %5d kimg | KID = %7.3fe-3 | best = %7.3fe-3 @ %s kimg | patience %d/%d%s",
                     cum, kid * 1e3, state["best_kid"] * 1e3, state["best_cum_kimg"],
                     state["bad_count"], g.patience, "  <-- tốt nhất" if improved else "")
            log.info("            %s", _last_tick_line(log_path))
            if state["bad_count"] >= g.patience:
                state.update(finished=True,
                             stop_reason=f"early stopping tại {cum} kimg (best {state['best_cum_kimg']} kimg)")
                self._save(state)
                return True
        return False
