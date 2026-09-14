"""Optional TensorBoard metric logging for policy training."""

import json
import logging
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import require_package

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


class TensorBoardLogger:
    def __init__(self, cfg: "TrainPipelineConfig", step: int = 0):
        require_package("tensorboard", extra="tensorboard")
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(cfg.tensorboard.log_dir or Path(cfg.output_dir) / "tensorboard")
        # A checkpoint at step N already includes N; only discard events after it.
        self._writer = SummaryWriter(log_dir=str(self.log_dir), purge_step=step + 1 if cfg.resume else None)
        try:
            self._writer.add_text("config", json.dumps(cfg.to_dict(), indent=2, default=str), step)
        except BaseException:
            self._writer.close()
            raise
        logging.info("TensorBoard logs: %s", self.log_dir)

    def log_dict(self, data: Mapping, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        def log_scalars(values: Mapping, prefix: str) -> None:
            for key, value in values.items():
                tag = f"{prefix}/{key}"
                if isinstance(value, Mapping):
                    log_scalars(value, tag)
                elif isinstance(value, Real):
                    self._writer.add_scalar(tag, value, step)

        log_scalars(data, mode)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
