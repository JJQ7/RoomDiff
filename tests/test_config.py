import pytest
import yaml

from roomdiff.config import Config, ConfigError, coerce, load_config, parse_override


def test_default_config_has_ingest_section():
    cfg = load_config()
    assert cfg.ingest.num_keyframes == 48
    assert cfg["ingest"]["max_side"] == 1024
    assert cfg.ingest.sharpness_window == 5
    assert cfg.ingest.min_motion == pytest.approx(0.02)
    assert cfg.ingest.jpeg_quality == 95
    for section in ("recon", "visibility", "instances", "changes", "semantics", "viz"):
        assert isinstance(cfg[section], Config)


def test_missing_attribute_raises_attribute_error():
    cfg = load_config()
    with pytest.raises(AttributeError):
        _ = cfg.ingest.no_such_key


def test_override_file_then_cli_overrides(tmp_path):
    override = tmp_path / "scene.yaml"
    override.write_text(yaml.safe_dump({"ingest": {"num_keyframes": 16, "max_side": 512}}))
    cfg = load_config(override_file=override, overrides=["ingest.max_side=640"])
    assert cfg.ingest.num_keyframes == 16  # from the file
    assert cfg.ingest.max_side == 640  # CLI wins over the file
    assert cfg.ingest.jpeg_quality == 95  # untouched default


@pytest.mark.parametrize(
    "text, value",
    [("32", 32), ("0.1", 0.1), ("1e-3", 0.001), ("true", True), ("False", False),
     ("null", None), ("chair", "chair"), (" 7 ", 7)],
)
def test_coerce(text, value):
    assert coerce(text) == value and type(coerce(text)) is type(value)


def test_parse_override_splits_and_types():
    assert parse_override("ingest.num_keyframes=32") == ("ingest.num_keyframes", 32)
    assert parse_override("a.prompt=a=b") == ("a.prompt", "a=b")  # split at the first '='


def test_int_is_accepted_for_a_float_key():
    cfg = load_config(overrides=["ingest.min_motion=0"])
    assert cfg.ingest.min_motion == 0.0 and isinstance(cfg.ingest.min_motion, float)


@pytest.mark.parametrize(
    "bad",
    ["ingest.num_keyframe=3",  # typo
     "nosection.key=1",  # unknown section
     "ingest=3",  # a section replaced by a value
     "ingest.max_side=big",  # wrong type
     "ingest.num_keyframes=4.5",
     "ingest.max_side=null",
     "ingest.max_side=auto",  # only an "auto" default accepts a number (not vice versa)
     "ingest.num_keyframes"],  # no '='
)
def test_bad_overrides_fail(bad):
    with pytest.raises(ConfigError):
        load_config(overrides=[bad])


def test_unknown_key_in_override_file_fails(tmp_path):
    override = tmp_path / "bad.yaml"
    override.write_text("ingest:\n  num_keyframe: 3\n")
    with pytest.raises(ConfigError, match=r"ingest\.num_keyframe"):
        load_config(override_file=override)


def test_to_dict_returns_plain_dicts():
    plain = load_config().to_dict()
    assert type(plain) is dict and type(plain["ingest"]) is dict
