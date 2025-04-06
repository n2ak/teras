import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self, inc, outc) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.LazyLinear(inc, 10),
            nn.LazyLinear(10, outc),
        )

    def forward(self, X,):
        X = self.seq(X)
        return X


def test_lightning():
    from examples.mnist import load_mnist
    X, y = load_mnist()
    import warnings
    warnings.filterwarnings("ignore")
    import teras.lightning

    model = CustomModel(64, 10)

    def optim_fns(model):
        return torch.optim.Adam(model.parameters()), None, None

    hist = teras.lightning.train(
        model,
        optim_fns,
        nn.functional.cross_entropy,
        X, y=y,
        validation_split=.2,
        epochs=20,
        train_batch_size=32,
        val_batch_size=32,
    )
    # pathname = "models/pathname.pt"
    # hist.save(pathname, model=model)
    # teras.load(pathname, model=model)  # .plot()
    # assert Path(pathname).exists()
