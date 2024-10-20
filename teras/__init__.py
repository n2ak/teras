from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import ignite.utils
import torch
import ignite
from torch import nn
import teras.metrics as M
import teras.callbacks as C
from teras.data import to_dataset
from collections import defaultdict


def assert_unique(arr: list, msg):
    def gen_name(elm):
        if isinstance(elm, (C.Callback, M.BaseMetric)):
            return elm._get_name()
        return type(elm)
    assert len({gen_name(l) for l in arr}) == len(arr), msg


class Model():
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device="cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        import tqdm
        self.bar: tqdm.tqdm = None

    def fit(
        self, X, y=None, *, epochs=2, metrics: list[M.BaseMetric] = [],
        validation_set=None,
        validation_split=None,
        batch_size=2, valid_batch_size=2,
        callbacks: list[C.Callback] = [],
        seed=1337,
        model_transform: lambda output: output,
        eval_output_transform=lambda x, y, y_pred: (y_pred, y),
        **kwargs,
    ):
        assert_unique(callbacks, "callbacks must be unique")
        assert_unique(metrics, "metrics must be unique")
        if seed:
            ignite.utils.manual_seed(seed)
        device = self.device
        model = self.model
        history = C.History()
        callbacks.extend([
            C.EpochBar(),
            # C.EpochLogger(),
            C.IterationLogger(),
            history,
        ])
        metrics.append(M.Loss(self.loss_fn))
        metrics = {m._get_name(): m for m in metrics}
        self.train_evaluator = create_supervised_evaluator(
            model, metrics=metrics, device=device,
            output_transform=eval_output_transform,
            model_transform=model_transform)
        self.epoch_state = defaultdict(list)

        self.trainer = create_supervised_trainer(
            model,
            self.optimizer,
            self.loss_fn,
            device,
            model_transform=model_transform,
            **kwargs,
        )
        self.train_dl, self.valid_dl = to_dataset(
            X, y, validation_set, validation_split,
            batch_size, valid_batch_size)
        do_valid = self.valid_dl is not None
        if do_valid:
            valid_metrics = {f"val_{k}": v for k, v in metrics.items()}
            self.valid_evaluator = create_supervised_evaluator(
                model, metrics=valid_metrics, device=device,
                output_transform=eval_output_transform,
                model_transform=model_transform)

        self.register_callbacks(callbacks)
        self.new_bar(1, len(self.train_dl))
        self.trainer.run(self.train_dl, max_epochs=epochs)
        return history.result()

    def new_bar(self, e, total):
        bar_format = None
        bar_format = '{desc}{percentage:3.0f}%|{bar:10} [{elapsed}<{remaining}] {postfix}'
        import tqdm
        self.bar = tqdm.trange(total, bar_format=bar_format)
        self.bar.set_description(f"Epoch {e:2}")

    def register_callbacks(self, callbacks: list[C.Callback]):
        for callback in callbacks:
            callback.set_context(self)
            for d in callback.called_on:
                actual_callback = callback
                if len(d) == 2:
                    target, event = d
                else:
                    target, event, actual_callback = d

                if target == "trainer":
                    engine = self.trainer
                elif target == "val":
                    engine = self.valid_evaluator
                elif target == "train":
                    engine = self.train_evaluator
                else:
                    raise Exception(f"Unknown target: {target}")
                engine.add_event_handler(event, actual_callback)
                # print(callback.__class__.__name__,
                #       "is listening for", event, "on", target)


def load(pathname, **modules): return C.TrainingResult.load(pathname, **modules)


def freeze(module: nn.Module):
    assert isinstance(module, nn.Module)
    for p in module.parameters():
        p.requires_grad = False
        p.grad = None


_events = [
    Events.EPOCH_COMPLETED,
    Events.EPOCH_STARTED,
    Events.STARTED,
    Events.COMPLETED,
    Events.ITERATION_STARTED,
    Events.ITERATION_COMPLETED,
    Events.EXCEPTION_RAISED,
    Events.GET_BATCH_STARTED,
    Events.GET_BATCH_COMPLETED,
    Events.DATALOADER_STOP_ITERATION,
    Events.TERMINATE,
    Events.TERMINATE_SINGLE_EPOCH,
    Events.INTERRUPT,
]


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
