import torch
from torch import nn
import torch.utils.data.dataset
import teras


def test_complex():
    import sklearn
    import sklearn.datasets
    import numpy as np
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    y = y.astype(np.longlong)
    pathname = "models/pathname.pt"

    class CustomModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq = nn.Sequential(
                nn.LazyLinear(10,),
                nn.LazyLinear(10,),
            )

        def forward(self, X):
            X = self.seq(X)

            return X, X.sum()
    model = CustomModel()
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

    hist.save(pathname, model=model)
    teras.load(pathname, model=model)  # .plot()
    from pathlib import Path
    assert Path(pathname).exists()


if __name__ == "__main__":
    test_complex()
