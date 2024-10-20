import torch
import __future__
from typing import TYPE_CHECKING
import ignite.metrics
import torch.optim.lr_scheduler as lrs

if TYPE_CHECKING:
    import teras
import ignite
import ignite.handlers
from ignite.engine import Events, Engine


class Callback:
    ties = False

    def _get_name(self):
        return self.__class__.__name__.lower()

    def set_context(self, context: "teras.Model"):
        self.context = context
        self.tied = True


def run_eval(evaluator: Engine, dl):
    evaluator.run(dl)
    metrics = evaluator.state.metrics
    return metrics


class EpochBar(Callback):
    called_on = [
        ("trainer", Events.EPOCH_COMPLETED),
    ]

    def __call__(self, engine: Engine):
        # if engine.should_terminate:
        #     return
        context = self.context
        do_valid = context.valid_dl is not None
        valid_metrics = {}
        train_metrics = run_eval(context.train_evaluator, context.train_dl)
        valid_log = ""

        def join(ms):
            return " ".join(
                [f"{name}: {value:.6f}" for name, value in ms.items()])
        if do_valid:
            valid_metrics = run_eval(context.valid_evaluator, context.valid_dl)
            # valid_metrics = {f'val_{k}': v for k, v in valid_metrics.items()}
            valid_log = " | " + join(valid_metrics)
        epoch = engine.state.epoch
        train_log = join(train_metrics)
        context.bar.set_postfix_str(
            f"{train_log}{valid_log}"
        )
        total = len(engine.state.dataloader)
        context.bar = None
        if epoch < engine.state.max_epochs:
            if not engine.should_terminate:
                context.new_bar(epoch+1, total)


class History(Callback):
    def __init__(self) -> None:
        from collections import defaultdict
        self.hist = defaultdict(list)
        self.called_on = [
            # ("trainer", Events.EPOCH_COMPLETED, self.on_epoch_completed),
            ("train", Events.COMPLETED, self.on_iteration_completed),
            ("val", Events.COMPLETED, self.on_val_completed),
        ]

    def on_epoch_completed(self, engine: Engine):
        pass

    def on_iteration_completed(self, engine: Engine):
        metrics = engine.state.metrics
        for k, v in metrics.items():
            assert isinstance(v, float), type(v)
            self.hist[k].append(v)

    def on_val_completed(self, engine: Engine):
        metrics = engine.state.metrics
        for k, v in metrics.items():
            assert isinstance(v, float), type(v)
            self.hist[k].append(v)

    def result(self):
        return TrainingResult(self.hist)


class TrainingResult:
    def __init__(self, hist: dict[str, list]) -> None:
        self.hist = hist
        self.metrics = [k for k in hist.keys() if not k.startswith("val_")]

    def plot(self):
        print(self.hist.keys())
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(self.metrics))
        axes = axes.flatten()

        def plot_ax(ax, name):
            assert name in self.hist
            ax.plot(self.hist[name], label=f"train_{name}")
            ax.set_title(name)
            name = "val_" + name
            if name in self.hist:
                ax.plot(self.hist[name], label=name)
            ax.legend()
        for ax, name in zip(axes, self.metrics):
            plot_ax(ax, name)
        fig.tight_layout()
        plt.show()

    def save(self, pathname, **modules: torch.nn.Module):
        import copy
        state = dict(
            hist=copy.deepcopy(self.hist),
            modules={
                k: v.state_dict() for k, v in modules.items()
            },
        )
        torch.save(state, pathname)
        print("State saved to", pathname)

    @staticmethod
    def load(pathname, **modules: torch.nn.Module):
        state: dict = torch.load(pathname)
        for k, v in modules.items():
            v.load_state_dict(state["modules"][k])
        print("State loaded from", pathname)
        return TrainingResult(hist=state["hist"])


class IterationLogger(Callback):
    called_on = [
        ("trainer", Events.ITERATION_STARTED),
    ]

    def __call__(self, engine: Engine):
        # if engine.should_terminate:
        #     return
        context = self.context
        itr = engine.state.iteration
        m = engine.state.metrics
        import numpy as np
        if context.bar is not None:
            context.bar.set_postfix({k: np.mean(v)
                                    for k, v in context.epoch_state})

        if context.bar is not None:
            # import time
            # time.sleep(.1)
            context.bar.update(1)


class ModelCheckpoint(Callback):
    called_on = [("val", Events.COMPLETED)]

    def __init__(
        self,
        dirname: str,
        monitor: str,
        direction="min",
        **kwargs,
    ):
        self.inner: ignite.handlers.EarlyStopping = None
        if monitor.startswith("val_"):
            self.called_on = [
                ("val", Events.COMPLETED),
            ]
        else:
            raise Exception("Only val accepted")
        self.monitor = monitor
        self.dirname = dirname
        self.direction = direction
        self.kwargs = kwargs

    def set_context(self, context: "teras.Model"):
        super().set_context(context)

        def score_function(engine: Engine):
            score = engine.state.metrics[self.monitor]
            if self.direction == "min":
                return -score
            elif self.direction == "max":
                return score
            else:
                raise Exception(f"Uknown direction: {self.direction}")
        self.inner = ignite.handlers.ModelCheckpoint(
            self.dirname,
            score_name=None,
            score_function=score_function,
            **self.kwargs,
        )

    def __call__(self, engine: Engine):
        self.inner.__call__(engine, {"model": self.context.model})


class LambdaCallback(Callback):
    called_on = []

    def __init__(
        self,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_train_begin=None,
        on_train_end=None,
        on_train_batch_begin=None,
        on_train_batch_end=None,
    ) -> None:
        if on_epoch_begin is not None:
            self.called_on.append(
                ("trainer", Events.EPOCH_STARTED, on_epoch_begin))
        if on_epoch_end is not None:
            self.called_on.append(
                ("trainer", Events.EPOCH_COMPLETED, on_epoch_end))
        if on_train_begin is not None:
            self.called_on.append(("trainer", Events.STARTED, on_train_begin))
        if on_train_end is not None:
            self.called_on.append(("trainer", Events.COMPLETED, on_train_end))
        if on_train_batch_begin is not None:
            self.called_on.append(
                ("trainer", Events.ITERATION_STARTED, on_train_batch_begin))
        if on_train_batch_end is not None:
            self.called_on.append(
                ("trainer", Events.ITERATION_COMPLETED, on_train_batch_end))


class EarlyStopping(Callback):

    def __init__(
        self,
        monitor: str,
        patience: int,
        min_delta: float = 0,
        cumulative_delta: bool = False,
        direction: str = "min",
    ):
        self.inner: ignite.handlers.EarlyStopping = None
        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.direction = direction
        if monitor.startswith("val_"):
            self.called_on = [
                ("val", Events.COMPLETED),
            ]
        else:
            raise Exception("Only val accepted")
        self.monitor = monitor

    def set_context(self, context: "teras.Model"):
        super().set_context(context)

        def score_function(engine: Engine):
            score = engine.state.metrics[self.monitor]
            if self.direction == "min":
                return -score
            elif self.direction == "max":
                return score
            else:
                raise Exception(f"Uknown direction: {self.direction}")
        self.inner = ignite.handlers.EarlyStopping(
            self.patience,
            score_function,
            context.trainer,
            self.min_delta,
            self.cumulative_delta,
        )

    def __call__(self, engine: Engine):
        self.inner.__call__(engine)


class LRScheduler(ignite.handlers.param_scheduler.LRScheduler, Callback):
    called_on = [
        ("trainer", Events.ITERATION_STARTED),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(self.inner_cls(*args, **kwargs))


class StepLR(LRScheduler):
    inner_cls = lrs.StepLR


class CyclicLR(LRScheduler):
    inner_cls = lrs.CyclicLR
