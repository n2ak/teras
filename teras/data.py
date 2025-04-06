
import torch
import torch.utils
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split


def to_ds(X, y, validation_set, validation_split):
    import numpy as np
    valid_dl = valid_ds = None
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if isinstance(X, torch.Tensor):
        assert isinstance(y, torch.Tensor)
        train_ds = TensorDataset(X, y)
    elif isinstance(X, Dataset):
        assert y is None
    else:
        raise ""
    if validation_set is not None:
        assert validation_split is None
        valid_dl = validation_set
    elif validation_split is not None:
        assert isinstance(validation_split, float)
        assert 0 < validation_split < 1
        train_ds, valid_ds = random_split(
            train_ds, [1-validation_split, validation_split])
    return train_ds, valid_ds


def get_dl(X, y, validation_set, validation_split, batch_size, valid_batch_size):
    train_ds, valid_ds = to_ds(X, y, validation_set, validation_split)
    if valid_ds is not None:
        valid_dl = DataLoader(
            valid_ds, batch_size=valid_batch_size)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
    )
    return train_dl, valid_dl
