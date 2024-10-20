
import torch
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
import ignite
import ignite.metrics
import ignite.metrics.metric


class BaseMetric():
    def _get_name(self):
        return self.__class__.__name__.lower()


class Loss(ignite.metrics.Loss, BaseMetric):
    pass


class Accuracy(ignite.metrics.Accuracy, BaseMetric):
    pass


class RMSE(ignite.metrics.RootMeanSquaredError, BaseMetric):
    pass


class MSE(ignite.metrics.MeanSquaredError, BaseMetric):
    pass


class MAE(ignite.metrics.MeanAbsoluteError, BaseMetric):
    pass


class CosineSimilarity(ignite.metrics.CosineSimilarity, BaseMetric):
    pass


class LabmdaMetric(ignite.metrics.Metric, BaseMetric):
    def __init__(
        self,
        func,
        device=torch.device("cpu"),
        skip_unrolling: bool = False,
    ):
        self.func = func
        super().__init__(
            lambda o: o,
            device,
            skip_unrolling=skip_unrolling,
        )

    def _get_name(self):
        # name = self.__class__.__name__
        name = self.func.__name__
        return name.lower()
    _state_dict_all_req_keys = ("_res", "_num_examples")

    @reinit__is_reduced
    def reset(self) -> None:
        self._res = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: list[torch.Tensor]) -> None:
        y_pred = output[0].detach()
        y = output[1].detach()
        self._res += self.func(y_pred, y,)
        self._num_examples += y.shape[0]

    @sync_all_reduce("_res", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise Exception("Need one example at least")
        return self._res.item() / self._num_examples
