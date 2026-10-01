"""`Layout`: NƠI DUY NHẤT định nghĩa đường dẫn artefact. Drive = lưu lâu dài; local (/content) = mất khi runtime reset.

Tên file giữ tương thích với notebook ISIC v5 / Brain Tumor v1 để dùng lại GAN và kết quả cũ trên Drive
(test `tests/regression/test_artifact_paths.py` khoá các đường dẫn này).
"""

from __future__ import annotations

from pathlib import Path

from .schema import Config


class Layout:
    def __init__(self, cfg: Config):
        p, d = cfg.paths, cfg.data
        s, sp, size = d.source, d.split, d.img_size

        drive = Path(p.drive_root)
        # Path("/a") / "/b" = "/b" -> đường dẫn tuyệt đối trong config được giữ nguyên
        self.drive_train_images = drive / s.train_images
        self.drive_train_labels = drive / s.train_labels if s.train_labels else None
        self.drive_test_images = drive / s.test_images if s.test_images else None
        self.drive_test_labels = drive / s.test_labels if s.test_labels else None
        self.split_file = drive / sp.file if sp.file else None
        self.expected_split = drive / sp.expected if sp.expected else None

        # ---- kết quả (Drive) ----
        self.results_dir = drive / f"results_{p.run_tag}"
        self.pred_dir = self.results_dir / "predictions"
        self.metrics_dir = self.results_dir / "metrics"          # số liệu thô (csv + json)
        self.tables_dir = self.results_dir / "tables"            # bảng cho bài báo (csv + json + tex)
        self.fingerprint_fig_dir = self.results_dir / "frequency_analysis"
        self.selection_fig_dir = self.results_dir / "selection_figures"
        self.gan_fig_dir = self.results_dir / "gan"
        self.run_manifest = self.results_dir / "run_manifest.json"

        # ---- checkpoint (Drive) ----
        self.gan_dir = drive / f"checkpoints_{p.gan_tag}" / "stylegan2ada"
        self.gan_state_json = self.gan_dir / "gan_state.json"
        self.gan_best_pkl = self.gan_dir / "best.pkl"
        self.gan_latest_pkl = self.gan_dir / "latest.pkl"
        self.gan_logs_dir = self.gan_dir / "logs_and_samples"
        self.clf_dir = drive / f"checkpoints_{p.run_tag}" / "classifiers"
        # E_d: của chính lần chạy, hoặc dùng lại của lần chạy khác (encoder.e_d_from_run)
        e = cfg.encoder
        self.e_d_run = {"": p.run_tag, "base": p.base_run_tag or p.run_tag}.get(e.e_d_from_run, e.e_d_from_run)
        self.e_d_ckpt = (drive / f"checkpoints_{self.e_d_run}" / "classifiers"
                         / f"Ed_{e.e_d_model}_s{e.e_d_seed}.weights.h5")
        self.data_dir = drive / f"checkpoints_{p.run_tag}" / "data"
        self.split_json = self.data_dir / "real_val_split.json"
        self.dataset_card = self.data_dir / "dataset_card.json"
        self.selections_json = self.data_dir / "selections.json"
        self.embeddings_npz = self.data_dir / "embeddings.npz"

        # ---- cục bộ (/content) ----
        local = Path(p.local_root)
        data_local = local / d.name          # mỗi bộ dữ liệu một thư mục riêng
        self.train_raw = data_local / "raw" / "train"
        self.test_raw = data_local / "raw" / "test"
        self.train_pp = data_local / f"pp{size}_png" / "train"
        self.test_pp = data_local / f"pp{size}_png" / "test"
        self.gan_dataset_zip = data_local / "gan_dataset" / f"train_cond_{size}.zip"
        self.gan_runs = local / "gan_runs"
        self.candidates = local / f"candidates_{p.run_tag}"
        self.variants = local / f"variants_{p.run_tag}"
        self.clf_ckpt = local / f"classifier_ckpt_{p.run_tag}"
        self.real_only = local / f"real_only_{p.run_tag}"

        # ---- repo ngoài ----
        ext = Path(p.external_root)
        self.sg2_repo = ext / "stylegan2-ada-pytorch"
        self.frank_repo = ext / "GANDCTAnalysis"

    def makedirs(self) -> None:
        for d in [self.results_dir, self.pred_dir, self.metrics_dir, self.tables_dir, self.gan_dir, self.clf_dir,
                  self.data_dir, self.train_raw, self.test_raw, self.train_pp, self.test_pp,
                  self.gan_dataset_zip.parent, self.gan_runs, self.candidates, self.variants, self.clf_ckpt,
                  self.real_only]:
            d.mkdir(parents=True, exist_ok=True)
