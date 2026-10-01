from pathlib import Path

import pytest

from isic_dass.config import Config, Layout, load_config, parse_override

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_dataclass_defaults():
    assert load_config(ROOT / "configs" / "default.yaml").to_dict() == Config().to_dict()


def test_smoke_yaml_loads():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    assert cfg.paths.run_tag == "smoke"
    assert cfg.classifier.seeds == [2026]


def test_overrides_are_typed():
    cfg = load_config(None, ["gan.batch=32", "classifier.ft_lr=1e-5", "classifier.seeds=[1, 2]",
                             "frequency.harmonize_channels=false"])
    assert cfg.gan.batch == 32
    assert cfg.classifier.ft_lr == pytest.approx(1e-5) and isinstance(cfg.classifier.ft_lr, float)
    assert cfg.classifier.seeds == [1, 2]
    assert cfg.frequency.harmonize_channels is False


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        load_config(None, ["gan.not_a_key=1"])
    with pytest.raises(ValueError):
        parse_override("gan.batch")


def test_layout_uses_tags():
    cfg = load_config(None, ["paths.drive_root=/d", "paths.gan_tag=g1", "paths.run_tag=r2"])
    layout = Layout(cfg)
    assert layout.gan_best_pkl == Path("/d/checkpoints_g1/stylegan2ada/best.pkl")
    assert layout.pred_dir == Path("/d/results_r2/predictions")
    assert layout.split_json == Path("/d/checkpoints_r2/data/real_val_split.json")
