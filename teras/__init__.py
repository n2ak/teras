from torch import nn
from teras.model import Model, load
import teras.metrics as M
import teras.callbacks as C


def freeze(module: nn.Module):
    assert isinstance(module, nn.Module)
    for p in module.parameters():
        p.requires_grad = False
        p.grad = None


def train(
    model,
    optimizer, loss_fn,
    X,
    y=None,
    metrics=[],
    callbacks=[],
    **kwargs,
):
    return Model(
        model, optimizer, loss_fn,
    ).fit(
        X, y,
        metrics=metrics,
        callbacks=callbacks,
        **kwargs
    )
