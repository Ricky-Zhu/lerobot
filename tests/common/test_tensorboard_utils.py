from contextlib import ExitStack
from types import SimpleNamespace

import draccus
import pytest

from lerobot.common.tensorboard_utils import TensorBoardLogger
from lerobot.configs.default import TensorBoardConfig


def make_config(tmp_path, *, resume=False, log_dir=None):
    return SimpleNamespace(
        tensorboard=TensorBoardConfig(enable=True, log_dir=log_dir),
        output_dir=tmp_path,
        resume=resume,
        to_dict=lambda: {"steps": 100},
    )


def read_events(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return EventAccumulator(str(path)).Reload()


def test_config_cli():
    assert not TensorBoardConfig().enable
    config = draccus.parse(TensorBoardConfig, args=["--enable=true", "--log_dir=/tmp/run"])
    assert config.enable
    assert config.log_dir == "/tmp/run"


def test_metrics_and_config_written(tmp_path):
    pytest.importorskip("tensorboard")
    logger = TensorBoardLogger(make_config(tmp_path))
    try:
        logger.log_dict({"loss": 0.5, "ema/decay": 0.99}, step=2)
        logger.log_dict(
            {"eval_loss": 0.25, "suite": {"pc_success": 75}, "video_paths": ["video.mp4"]},
            step=2,
            mode="eval",
        )
        # Events are readable during training, before close().
        events = read_events(tmp_path / "tensorboard")
        assert events.Scalars("train/loss")[0].value == 0.5
        assert events.Scalars("eval/eval_loss")[0].step == 2
        assert events.Scalars("eval/suite/pc_success")[0].value == 75
        assert "eval/video_paths" not in events.Tags()["scalars"]
        assert "config/text_summary" in events.Tags()["tensors"]
        with pytest.raises(ValueError):
            logger.log_dict({"loss": 1}, step=3, mode="invalid")
    finally:
        logger.close()


def test_resume_preserves_checkpoint_step_and_purges_future(tmp_path):
    pytest.importorskip("tensorboard")
    log_dir = str(tmp_path / "custom")
    logger = TensorBoardLogger(make_config(tmp_path, log_dir=log_dir))
    for step in (5, 10, 15):
        logger.log_dict({"loss": step}, step)
    logger.close()

    resumed = TensorBoardLogger(make_config(tmp_path, resume=True, log_dir=log_dir), step=10)
    resumed.log_dict({"loss": 0.5}, step=12)
    resumed.close()
    events = read_events(log_dir).Scalars("train/loss")
    assert [(event.step, event.value) for event in events] == [(5, 5), (10, 10), (12, 0.5)]


def test_writer_closed_on_exception(tmp_path, monkeypatch):
    pytest.importorskip("tensorboard")
    from unittest.mock import MagicMock

    writer = MagicMock()
    monkeypatch.setattr("torch.utils.tensorboard.SummaryWriter", lambda **kwargs: writer)
    with pytest.raises(RuntimeError, match="training failed"), ExitStack() as cleanup:
        logger = TensorBoardLogger(make_config(tmp_path))
        cleanup.callback(logger.close)
        raise RuntimeError("training failed")
    writer.close.assert_called_once()


def test_missing_dependency_has_install_hint(tmp_path, monkeypatch):
    from lerobot.utils import import_utils

    monkeypatch.setattr(import_utils, "is_package_available", lambda *args: False)
    monkeypatch.setattr(import_utils, "_require_package_cache", {})
    with pytest.raises(ImportError, match=r"lerobot\[tensorboard\]"):
        TensorBoardLogger(make_config(tmp_path))
