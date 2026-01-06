from typing import override

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class ModelCheckpointWithMinBatches(ModelCheckpoint):
    def __init__(self, min_batches: int = 0, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.min_batches: int = min_batches
        self._batches_seen: int = 0

    @override
    def on_train_batch_end(
        self,
        trainer,  # noqa: ANN001
        pl_module,  # noqa: ANN001
        outputs,  # noqa: ANN001
        batch,  # noqa: ANN001
        batch_idx,  # noqa: ANN001
    ) -> None:
        self._batches_seen += 1

    @override
    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if self._batches_seen >= self.min_batches:
            super().on_validation_end(trainer, pl_module)
        # else: skip updating best model


class EarlyStoppingWithMinBatches(EarlyStopping):
    """Add min epoch functionality to EarlyStopping."""

    def __init__(self, min_batches: int = 0, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.min_batches: int = min_batches
        self._batches_seen: int = 0

    @override
    def on_train_batch_end(
        self,
        trainer,  # noqa: ANN001
        pl_module,  # noqa: ANN001
        outputs,  # noqa: ANN001
        batch,  # noqa: ANN001
        batch_idx,  # noqa: ANN001
    ) -> None:
        self._batches_seen += 1

    @override
    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if self._batches_seen >= self.min_batches:
            super().on_validation_end(trainer, pl_module)
        # else: do nothing, don't check for early stopping yet
