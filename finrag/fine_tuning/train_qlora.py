"""Compatibility entry point; implementation lives in fine_tuning.training."""

from .training.train_qlora import main, train

__all__ = ["main", "train"]


if __name__ == "__main__":
    raise SystemExit(main())
