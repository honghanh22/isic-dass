import dataclasses
from pathlib import Path

import pytest
from conftest import CONFIGS

from dass.config import Config, Layout, load_config, parse_override
from dass.config.loader import apply_file

BASES = sorted((CONFIGS / "_base_").glob("*.yaml"))
EXPERIMENTS = ["isic2016_dass.yaml", "rsna_pneumonia_dass.yaml", "pcam_dass.yaml"]


def test_base_yaml_matches_schema_defaults():
    """configs/_base_/*.yaml và giá trị mặc định trong schema phải trùng nhau (một nguồn sự thật)."""
    assert load_config(BASES, check=False).to_dict() == Config().to_dict()


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiments_load_and_validate(name):
    cfg = load_config([CONFIGS / "experiments" / name])
    assert cfg.data.name and cfg.paths.drive_root
    # công thức chuẩn hoá (ISIC v5), giống nhau cho mọi bộ dữ liệu
    assert (cfg.selection.lambda_v, cfg.selection.gamma, cfg.classifier.monitor) == (1.0, 0.5, "val_auc")
    assert cfg.evaluation.threshold == 0.5














def test_rsna_config_and_augment_profiles():
    rsna = load_config([CONFIGS / "experiments" / "rsna_pneumonia_dass.yaml"])
    assert rsna.data.source.type == "dicom_csv" and rsna.data.channels == 1 and rsna.data.augment_profile == "upright"
    assert (rsna.data.source.id_column, rsna.data.source.label_column) == ("patientId", "Target")
    assert rsna.paths.gan_tag == "rsna" and rsna.data.split.type == "stratified"
    isic = load_config([CONFIGS / "experiments" / EXPERIMENTS[0]])
    assert isic.data.augment_profile == "rotation_invariant"            # mặc định: hành vi cũ của ISIC
    assert rsna.selection == isic.selection and rsna.encoder == isic.encoder        # cùng công thức DASS / E_d
    assert rsna.classifier == isic.classifier                            # cùng giao thức classifier (1.11.0)
    # đánh giá theo tư thế chụp (đăng ký trước) chỉ ở RSNA; ISIC không có metadata
    assert rsna.evaluation.subgroup_columns == ["ViewPosition"] and isic.evaluation.subgroup_columns == []
    assert dataclasses.replace(rsna.evaluation, subgroup_columns=[]) == isic.evaluation
    # GAN: chỉ khác mirror (X-quang ngực không đối xứng trái / phải), mọi tham số khác giống ISIC
    assert isic.generator.mirror is True and rsna.generator.mirror is False
    assert dataclasses.replace(rsna.generator, mirror=True) == isic.generator
    with pytest.raises(ValueError, match="augment_profile"):
        load_config([CONFIGS / "experiments" / "rsna_pneumonia_dass.yaml"], ["data.augment_profile=random"])
    with pytest.raises(ValueError, match="dicom_csv"):
        load_config([CONFIGS / "experiments" / "rsna_pneumonia_dass.yaml"], ["data.source.id_column=''"])


@pytest.mark.parametrize("k", ["0.5", "0.15", "-1"])
def test_pool_mult_below_one_rejected(k):
    with pytest.raises(ValueError, match="pool_mult"):
        load_config([CONFIGS / "experiments" / EXPERIMENTS[0]], [f"selection.pool_mult={k}"])


def test_bad_comparison_rejected():
    with pytest.raises(ValueError, match="comparisons"):
        load_config([CONFIGS / "experiments" / EXPERIMENTS[0]], ["evaluation.comparisons=[[M6_dass]]"])





def test_unknown_backbone_rejected_at_load_time():
    with pytest.raises(ValueError, match="vgg16"):
        load_config([CONFIGS / "experiments" / EXPERIMENTS[0]], ["classifier.models=[ResNet50, vgg16]"])
    with pytest.raises(ValueError, match="seeds"):
        load_config([CONFIGS / "experiments" / EXPERIMENTS[0]], ["classifier.seeds=[]"])


def test_experiments_differ_only_in_allowed_sections():
    """ISIC vs RSNA: khác ở dữ liệu / đường dẫn, cộng đúng các ngoại lệ đã ghim trong test_rsna_config_and_augment_profiles
    (generator.mirror, evaluation.subgroup_columns); DASS / E_d / classifier giống hệt."""
    a = load_config([CONFIGS / "experiments" / EXPERIMENTS[0]]).to_dict()
    b = load_config([CONFIGS / "experiments" / EXPERIMENTS[1]]).to_dict()
    assert {k for k in a if a[k] != b[k]} == {"paths", "data", "generator", "evaluation"}


def test_k_is_identical_across_datasets():
    """k (pool_mult) và mọi tham số DASS phải giống nhau giữa các bộ dữ liệu; k = 2 (2.0.0)."""
    a = load_config([CONFIGS / "experiments" / EXPERIMENTS[0]])
    b = load_config([CONFIGS / "experiments" / EXPERIMENTS[1]])
    assert a.selection == b.selection and a.encoder == b.encoder
    assert a.selection.pool_mult == b.selection.pool_mult == 2.0
    for ds in (CONFIGS / "datasets").glob("*.yaml"):          # không file dataset nào được ghi đè tham số chọn ảnh
        text = ds.read_text(encoding="utf-8")
        assert "selection:" not in text and "pool_mult" not in text and "encoder:" not in text, ds.name


def test_e_d_reuse_from_main_run():
    cfg = load_config([CONFIGS / "experiments" / EXPERIMENTS[1]], ["encoder.e_d_from_run=base"], tag="x")
    lay = Layout(cfg)
    assert lay.e_d_run == "rsna_v5" and lay.results_dir.name == "results_rsna_v5_x"
    assert lay.e_d_ckpt == Path(cfg.paths.drive_root) / "checkpoints_rsna_v5" / "classifiers" / \
        "Ed_DenseNet121_s4242.weights.h5"
    with pytest.raises(ValueError, match="--tag"):          # "base" mà không có --tag -> lỗi
        load_config([CONFIGS / "experiments" / EXPERIMENTS[1]], ["encoder.e_d_from_run=base"])


def test_main_run_trains_its_own_e_d():
    lay = Layout(load_config([CONFIGS / "experiments" / EXPERIMENTS[1]]))
    assert lay.e_d_run == "rsna_v5" and lay.e_d_ckpt.parent == lay.clf_dir


def test_smoke_profile_composes_with_any_dataset():
    for name in EXPERIMENTS:
        cfg = load_config([CONFIGS / "experiments" / name, CONFIGS / "experiments" / "smoke.yaml"])
        assert cfg.paths.run_tag == "smoke" and cfg.classifier.seeds == [2026]


def test_overrides_and_tag():
    cfg = load_config([CONFIGS / "experiments" / EXPERIMENTS[0]],
                      ["generator.batch=32", "classifier.ft_lr=1e-5", "classifier.seeds=[1, 2]"], tag="abl")
    assert cfg.generator.batch == 32 and isinstance(cfg.classifier.ft_lr, float)
    assert cfg.classifier.seeds == [1, 2] and cfg.paths.run_tag == "v14_abl"


def test_unknown_key_and_bad_override():
    with pytest.raises(KeyError):
        load_config([CONFIGS / "experiments" / EXPERIMENTS[0]], ["generator.not_a_key=1"])
    with pytest.raises(ValueError):
        parse_override("generator.batch")


def test_base_inheritance_and_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\nseed: 1\n")
    (tmp_path / "b.yaml").write_text("seed: 2\ndeterministic: true\n")
    cfg = Config()
    apply_file(cfg, tmp_path / "a.yaml")
    assert cfg.seed == 1 and cfg.deterministic is True        # file con ghi đè file cha
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ValueError, match="vòng tròn"):
        apply_file(Config(), tmp_path / "a.yaml")


def test_dict_values_are_replaced_not_merged(tmp_path):
    (tmp_path / "x.yaml").write_text("data: {classes: {a: 0, b: 1}}\n")
    (tmp_path / "y.yaml").write_text("data: {classes: {c: 0, d: 1}}\n")
    cfg = load_config([tmp_path / "x.yaml", tmp_path / "y.yaml"], check=False)
    assert cfg.data.classes == {"c": 0, "d": 1}


@pytest.mark.parametrize("override", [
    "data.classes={a: 0, b: 1, c: 2}", "data.classes={a: 1, b: 2}", "data.channels=2",
    "data.source.type=zip", "data.preprocess.resize=crop", "data.split.type=kfold",
    "data.split.test=0.2",                       # ISIC có test riêng -> holdout_val không nhận split.test
    "data.split.type=stratified",                 # stratified cần nguồn KHÔNG có test riêng
])
def test_invalid_config_rejected(override):
    with pytest.raises(ValueError):
        load_config([CONFIGS / "experiments" / EXPERIMENTS[0]], [override])


def test_layout_paths_relative_and_absolute():
    cfg = load_config([CONFIGS / "experiments" / EXPERIMENTS[0]], ["data.source.train_images=/abs/imgs"])
    layout = Layout(cfg)
    assert layout.drive_train_images == Path("/abs/imgs")
    assert layout.drive_test_images.parent == Path(cfg.paths.drive_root)



def test_run_chain_trains_one_seed_per_process():
    from dass.cli import train_commands

    cmds = train_commands(["py", "-m", "dass"], ["EfficientNetV2B0", "ResNet50"], [2026, 2027, 2028])
    assert len(cmds) == 6
    assert cmds[0][-4:] == ["--model", "EfficientNetV2B0", "--seeds", "2026"]
    assert cmds[2][-4:] == ["--model", "EfficientNetV2B0", "--seeds", "2028"]
    assert cmds[3][-4:] == ["--model", "ResNet50", "--seeds", "2026"]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_design_2_0(name):
    """Thiết kế 2.0.0 (người dùng quyết định): B0–B3 + Balanced GAN (ảnh sinh chỉ ở lớp thiểu số, cân bằng 1 : 1) M1–M6,
    k = 2; họ chính H1 DASS vs Random, H2 DASS vs ROS, H3 DASS vs Imbalanced; 3 backbone (cùng head), 3 seed."""
    cfg = load_config([CONFIGS / "experiments" / name])
    s, ev, clf = cfg.selection, cfg.evaluation, cfg.classifier
    assert (s.pool_mult, s.sim_topk, s.lambda_v, s.lambda_d, s.alpha, s.beta, s.gamma) == (2.0, 5, 1, 1, 1, 1, 0.5)
    assert ev.baseline_method == "B0_imbalanced"
    assert ev.primary_comparisons == [["M6_dass", "M1_random"], ["M6_dass", "B2_ros"], ["M6_dass", "B0_imbalanced"]]
    assert clf.models == ["ResNet50", "ConvNeXtTiny", "SwinT"] and clf.seeds == [2026, 2027, 2028]
    assert clf.swin_preset == "swin_tiny_patch4_window7_224"            # tên preset KerasHub chính thức


def test_result_tables():
    """Hai bảng của bài: so sánh (B0–B3, B4 = M1, B5 = M6) và ablation (M1–M6); mọi mã đều có tên hiển thị chung."""
    from dass.selection import METHODS, SELECTION_METHODS

    ev = load_config([CONFIGS / "experiments" / EXPERIMENTS[0]]).evaluation
    assert list(ev.method_labels) == list(METHODS)
    assert list(ev.result_tables["comparison"]) == ["B0_imbalanced", "B1_class_weight", "B2_ros", "B3_augment",
                                                    "M1_random", "M6_dass"]
    assert list(ev.result_tables["ablation"]) == list(SELECTION_METHODS)
    assert ev.result_tables["comparison"]["M6_dass"].startswith("B5:")


@pytest.mark.parametrize("ds,run,ed", [("isic2016", "v14_srv", ""), ("rsna_pneumonia", "rsna_v5_srv", ""),
                                       ("pcam", "pcam6_v1_srv", "")])
def test_server_configs(ds, run, ed):
    """Server: run_tag mới của thiết kế 2.0.0; ISIC / RSNA train lại E_d, PCam dùng CHÍNH E_d đã chọn snapshot GAN."""
    cfg = load_config([CONFIGS / "server" / f"{ds}.yaml"], tag="srv")
    assert cfg.paths.run_tag == run and cfg.encoder.e_d_from_run == ed
    assert cfg.classifier.models == ["ResNet50", "ConvNeXtTiny", "SwinT"]


@pytest.mark.parametrize("ds", ["rsna_pneumonia", "isic2016"])
def test_colab_config_matches_server_except_paths(ds):
    """configs/colab/*: cấu hình server, chỉ khác đường dẫn Drive / HDD và hậu tố _srv của run_tag."""
    colab = load_config([CONFIGS / "colab" / f"{ds}.yaml"]).to_dict()
    server = load_config([CONFIGS / "server" / f"{ds}.yaml"], tag="srv").to_dict()
    assert {k for k in colab if k not in ("paths", "data") and colab[k] != server[k]} == set()
    assert {k for k in colab["data"] if colab["data"][k] != server["data"][k]} == {"source"}
    assert server["paths"]["run_tag"] == colab["paths"]["run_tag"] + "_srv"
