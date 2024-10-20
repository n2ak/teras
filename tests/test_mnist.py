import torch
import torch.utils.data.dataset
import teras


def test_mnist():
    import sklearn.datasets
    import sklearn
    import numpy as np
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    model1 = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, len(np.unique(y))),
    )

    optimizer = torch.optim.Adam(model1.parameters())
    model = teras.Model(
        model1, optimizer, torch.nn.functional.cross_entropy
    )
    hist = model.fit(
        X, y.astype(np.longlong),
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
        # scaler=True,
        # amp_mode='amp',
    )
    pathname = "models/pathname.pt"
    hist.save(pathname, model=model1)
    teras.load(pathname, model=model1)  # .plot()
    from pathlib import Path
    assert Path(pathname).exists()


if __name__ == "__main__":
    test_mnist()
