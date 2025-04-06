import torch
import teras
from torch import nn
from pathlib import Path
from examples.mnist import load_mnist


class CustomModel(nn.Module):
    def __init__(self, inc, outc) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(inc, 32),
            nn.Linear(32, outc),
        )

    def forward(self, X):
        X = self.seq(X)
        return X, X.sum()


def test_complex():
    X, y = load_mnist()

    model = CustomModel(64, 10)
    optimizer = torch.optim.Adam(model.parameters())

    def model_transform(X):
        X, _ = X
        return X

    hist = teras.train(
        model, optimizer, torch.nn.functional.cross_entropy,
        X, y=y,
        metrics=[
            teras.M.Accuracy(),
        ],
        callbacks=[
            teras.C.ModelCheckpoint(
                "models",
                "val_loss",
                require_empty=False,
            ),
            teras.C.StepLR(optimizer, 1),
            teras.C.EarlyStopping("val_loss", 2,),
            teras.C.LambdaCallback(
                on_train_begin=lambda: print("Starting"),
                on_train_end=lambda: print("Ending"),
            ),
        ],
        validation_split=.2,
        epochs=20,
        batch_size=32,
        valid_batch_size=32,
        model_transform=model_transform,
    )
    pathname = "models/pathname.pt"
    hist.save(pathname, model=model)
    teras.load(pathname, model=model)  # .plot()
    assert Path(pathname).exists()
