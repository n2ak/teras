import torch
import lightning as L
from typing import Callable
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LRScheduler


class Model(L.LightningModule):
    def __init__(self, model, config: "Config"):
        super().__init__()
        self.model = model
        self.config = config

    def _forward(self, batch, _):
        X, y = batch
        logits = self.model(X)
        loss = self.config.loss_fn(logits, y)
        return logits, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._forward(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self._forward(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizers = []
        for optim, lr_scheduler, monitor in [fn(self) for fn in self.config.optim_fns]:
            if lr_scheduler is not None:
                lr_scheduler = lr_scheduler(optim)
            optimizers.append(dict(
                optimizer=optim,
            ))
            if lr_scheduler:
                optimizers[0]["lr_scheduler"] = dict(
                    scheduler=lr_scheduler,
                    monitor=monitor,
                )
        if len(optimizers) == 1:
            optimizers = optimizers[0]
        return optimizers

    def train_dataloader(self):
        return DataLoader(
            self.config.train_dataset,
            collate_fn=self.config.train_collate_fn,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.config.val_dataset,
            collate_fn=self.config.eval_collate_fn,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


OptimFn = Callable[[L.LightningModule], tuple[Optimizer, LRScheduler, str]]


@dataclass
class Config:
    train_dataset: Dataset
    val_dataset: Dataset
    train_collate_fn: any
    eval_collate_fn: any
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    optim_fns: list[OptimFn]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
