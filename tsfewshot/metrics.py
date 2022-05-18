from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
from numba import jit, prange
from torch import nn

from tsfewshot.config import Config


class BaseMetric(ABC, nn.Module):
    """Abstract base metric class.

    Calculates metric on the last ``cfg.predict_last_n`` time steps. (All steps if ``cfg.predict_last_n is None``)

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target_vars : List[str]
        List of target variables.
    timeseries_median : bool, optional
        If True, will calculate the metric for each timeseries and then the median of these metrics.
        Otherwise, will calculate the metric for all timeseries combined.
        Must provide `timeseries_id` if True (this identifies the timeseries that each entry belongs to).
    """

    def __init__(self, cfg: Config, target_vars: List[str], timeseries_median: bool = False):
        super().__init__()
        self._metric_name: str
        self._target_vars = target_vars
        self._predict_last_n = cfg.predict_last_n

        self._timeseries_median = timeseries_median

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor, dataset: str,
                timeseries_id: List[int] = None) -> Dict[str, torch.Tensor]:
        assert prediction.shape == target.shape
        if self._predict_last_n is not None:
            prediction, target = prediction[:, -self._predict_last_n:], target[:, -self._predict_last_n:]

        metrics = {}
        if self._timeseries_median:
            if self._metric_name != 'rmse':
                # we've only implemented the numba function for rmse
                raise NotImplementedError('timeseries_median only implemented for RMSE.')
            if timeseries_id is None:
                raise ValueError('Must provide timeseries_id if median is True.')

            per_target_metrics, global_metrics = numba_per_ts_rmse(prediction.cpu().numpy(),
                                                                   target.cpu().numpy(),
                                                                   np.array(timeseries_id),
                                                                   np.array(self._target_vars))

            for target_name in self._target_vars:
                metrics[f'{target_name}_{self._metric_name}_median'] = np.nanmedian(per_target_metrics[target_name])
            metrics[f'{self._metric_name}_median'] = np.nanmedian(global_metrics)

        else:
            # per-target metrics
            for i, target_name in enumerate(self._target_vars):
                metrics[f'{target_name}_{self._metric_name}'] = self._metric_fn(prediction[:, :, i],  # type: ignore
                                                                                target[:, :, i],
                                                                                dataset=dataset).item()
            # global metric
            metrics[self._metric_name] = self._metric_fn(prediction.transpose(1, 2).reshape(prediction.shape[0]
                                                                                            * prediction.shape[2],
                                                                                            prediction.shape[1]),
                                                         target.transpose(1, 2).reshape(target.shape[0]
                                                                                        * target.shape[2],
                                                                                        target.shape[1]),
                                                         dataset=dataset).item()

        return metrics

    @abstractmethod
    def _metric_fn(self, prediction: torch.Tensor, target: torch.Tensor, dataset: str) -> torch.Tensor:
        pass


class MAEMetric(BaseMetric):
    """Mean Absolute Error metric.

    Calculates MAE on the last ``cfg.predict_last_n`` time steps. (All steps if ``cfg.predict_last_n is None``)

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target_vars : List[str]
        List of target variables.
    """

    def __init__(self, cfg: Config, target_vars: List[str]):
        super().__init__(cfg, target_vars)
        self._metric_name = 'mae'
        self._loss = nn.L1Loss()

    def _metric_fn(self, prediction: torch.Tensor, target: torch.Tensor, dataset: str) -> torch.Tensor:
        return self._loss(prediction, target)


class MSEMetric(BaseMetric):
    """Mean Squared Error metric.

    Calculates MSE on the last ``cfg.predict_last_n`` time steps. (All steps if ``cfg.predict_last_n is None``)

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target_vars : List[str]
        List of target variables.
    """

    def __init__(self, cfg: Config, target_vars: List[str]):
        super().__init__(cfg, target_vars)
        self._metric_name = 'mse'
        self._loss = nn.MSELoss()

    def _metric_fn(self, prediction: torch.Tensor, target: torch.Tensor, dataset: str) -> torch.Tensor:
        return self._loss(prediction, target)


class RMSEMetric(BaseMetric):
    """Root Mean Squared Error metric.

    Calculates RMSE on the last ``cfg.predict_last_n`` time steps. (All steps if ``cfg.predict_last_n is None``)

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target_vars : List[str]
        List of target variables.
    timeseries_median : bool, optional
        If True, will calculate the metric for each timeseries and then the median of these metrics.
        Otherwise, will calculate the metric for all timeseries combined.
        Must provide `timeseries_id` if True (this identifies the timeseries that each entry belongs to).
    """

    def __init__(self, cfg: Config, target_vars: List[str], timeseries_median: bool = False):
        super().__init__(cfg, target_vars, timeseries_median=timeseries_median)
        self._metric_name = 'rmse'
        self._loss = nn.MSELoss()

    def _metric_fn(self, prediction: torch.Tensor, target: torch.Tensor, dataset: str) -> torch.Tensor:
        return torch.sqrt(self._loss(prediction, target))


class NSEMetric(BaseMetric):
    """Nash--Sutcliffe Efficiency [#]_ metric.

    Calculates NSE on the last ``cfg.predict_last_n`` time steps. (All steps if ``cfg.predict_last_n is None``).
    Only supports single-target predictions.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target_vars : List[str]
        List of target variables.
    stds : Dict[str, torch.Tensor]
        Standard deviations of the target variable for each dataset.

    References
    ----------
    .. [#] Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting through conceptual models part I - A
        discussion of principles". Journal of Hydrology. 10 (3): 282-290. doi:10.1016/0022-1694(70)90255-6.
    """

    def __init__(self, cfg: Config, target_vars: List[str], stds: Dict[str, torch.Tensor]):
        super().__init__(cfg, target_vars)
        self._stds = stds
        self._metric_name = 'nse'

        if len(target_vars) > 1:
            raise ValueError('NSE metric only supports single-target prediction tasks.')

    def _metric_fn(self, prediction: torch.Tensor, target: torch.Tensor, dataset: str) -> torch.Tensor:

        denominator = (target.shape[0] * target.shape[1]) * (self._stds[dataset]**2)
        numerator = ((prediction - target)**2).sum()

        value = 1 - numerator / denominator

        return value


class AccuracyMetric(BaseMetric):
    """Accuracy metric.

    Calculates classification accuracy, only supported for single-step prediction, i.e., ``cfg.predict_last_n == 1``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target_vars : List[str]
        List of target variables.
    timeseries_median : bool, optional
        If True, will calculate the metric for each timeseries and then the median of these metrics.
        Otherwise, will calculate the metric for all timeseries combined.
        Must provide `timeseries_id` if True (this identifies the timeseries that each entry belongs to).
    """

    def __init__(self, cfg: Config, target_vars: List[str], timeseries_median: bool = False):
        if cfg.predict_last_n > 1:
            raise ValueError('Accuracy only supported for single-step prediction.')

        super().__init__(cfg, target_vars, timeseries_median=timeseries_median)
        self._metric_name = 'accuracy'

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor, dataset: str,
                timeseries_id: List[int] = None) -> Dict[str, torch.Tensor]:
        # last dim in prediction is n_classes, last dim in target is 1
        assert prediction.shape[:-1] == target.shape[:-1]

        # different classes are different integers within dim 2, not different dimensions
        assert target.shape[2] == 1

        # can directly choose -1 in dim 1, because we only allow predict_last_n == 1.
        # dim 2 in target will always have shape 1, because classes are different integers in that dimension.
        prediction, target = prediction[:, -1], target[:, -1, 0]

        # global metric only. per-target metric doesn't work for classification
        metrics = {self._metric_name: self._metric_fn(prediction, target, dataset=dataset).item()}

        return metrics  # type: ignore

    def _metric_fn(self, prediction: torch.Tensor, target: torch.Tensor, dataset: str) -> torch.Tensor:

        # could probably relax this condition to other int types or bool if needed
        assert target.dtype == torch.long

        if prediction.shape[1] == 1:
            # binary classification
            class_prediction = prediction >= 0.5
        else:
            # multiclass prediction
            class_prediction = torch.argmax(prediction, dim=1)

        correct = class_prediction.to(dtype=target.dtype) == target

        return correct.sum() / target.shape[0]


class CrossentropyMetric(BaseMetric):
    """Crossentropy metric.

    Only supported for single-step prediction, i.e., ``cfg.predict_last_n == 1``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target_vars : List[str]
        List of target variables.
    timeseries_median : bool, optional
        If True, will calculate the metric for each timeseries and then the median of these metrics.
        Otherwise, will calculate the metric for all timeseries combined.
        Must provide `timeseries_id` if True (this identifies the timeseries that each entry belongs to).
    """

    def __init__(self, cfg: Config, target_vars: List[str], timeseries_median: bool = False):
        if cfg.predict_last_n > 1:
            raise ValueError('CrossentropyMetric only supported for single-step prediction.')

        super().__init__(cfg, target_vars, timeseries_median=timeseries_median)
        self._metric_name = 'crossentropy'
        self._cross_entropy = nn.CrossEntropyLoss()

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor, dataset: str,
                timeseries_id: List[int] = None) -> Dict[str, torch.Tensor]:
        # last dim in prediction is n_classes, last dim in target is 1
        assert prediction.shape[:-1] == target.shape[:-1]

        # different classes are different integers within dim 2, not different dimensions
        assert target.shape[2] == 1

        # can directly choose -1 in dim 1, because we only allow predict_last_n == 1.
        # dim 2 in target will always have shape 1, because classes are different integers in that dimension.
        prediction, target = prediction[:, -1], target[:, -1, 0]

        # global metric only. per-target metric doesn't work for classification
        metrics = {self._metric_name: self._metric_fn(prediction, target, dataset=dataset).item()}

        return metrics  # type: ignore

    def _metric_fn(self, prediction: torch.Tensor, target: torch.Tensor, dataset: str) -> torch.Tensor:
        # could probably relax this condition to other int types or bool if needed
        assert target.dtype == torch.long

        return self._cross_entropy(prediction, target)


@jit(nopython=True, parallel=True)  # type: ignore
def numba_per_ts_rmse(prediction: np.ndarray, target: np.ndarray, timeseries_id: np.ndarray, target_vars: np.ndarray):
    """Calculate RMSE for each individual timeseries. """
    n_timeseries = int(max(timeseries_id)) + 1

    per_target_metrics = {}
    for target_name in target_vars:
        per_target_metrics[target_name] = np.full(n_timeseries, np.nan)

    global_metrics = np.full(n_timeseries, np.nan)
    for j in prange(n_timeseries):
        mask = timeseries_id == j
        if not np.any(mask):
            continue
        timeseries_pred = prediction[mask]
        timeseries_target = target[mask]

        # per-target metrics
        for i, target_name in enumerate(target_vars):
            per_target_metrics[target_name][j] = np.sqrt(np.mean(np.square(timeseries_pred[:, :, i]
                                                                           - timeseries_target[:, :, i])))

        # global metric: merge feature and batch dimensions
        ts_pred = np.ascontiguousarray(np.transpose(timeseries_pred, (0, 2, 1))) \
            .reshape(timeseries_pred.shape[0] * timeseries_pred.shape[2], timeseries_pred.shape[1])
        ts_target = np.ascontiguousarray(np.transpose(timeseries_target, (0, 2, 1))) \
            .reshape(timeseries_target.shape[0] * timeseries_target.shape[2], timeseries_target.shape[1])
        global_metrics[j] = np.sqrt(np.mean(np.square(ts_pred - ts_target)))
    return per_target_metrics, global_metrics


def get_metrics(cfg: Config, target_variables: List[str], stds: Optional[Dict[str, torch.Tensor]]) -> List[BaseMetric]:
    """Return modules that calculate the specified metrics.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    target_vars : List[str]
        List of target variables.
    stds : Dict[str, torch.Tensor]
        Standard deviations of the target variable for each dataset. Required for NSEMetric.

    Returns
    -------
    List[BaseMetric]
        Metric functions
    """
    metrics = []
    for metric in cfg.metric:
        if metric == 'mse':
            metrics.append(MSEMetric(cfg, target_variables))
        elif metric in ['rmse', 'rmse_median']:
            # metric with _median calculates the median of the metrics for each individual timeseries
            metrics.append(RMSEMetric(cfg, target_variables, timeseries_median=metric.endswith('_median')))
        elif metric == 'nse':
            if stds is None:
                raise ValueError('stds is required for NSE metric.')
            metrics.append(NSEMetric(cfg, target_variables, stds=stds))
        elif metric == 'mae':
            metrics.append(MAEMetric(cfg, target_variables))
        elif metric == 'accuracy':
            metrics.append(AccuracyMetric(cfg, target_variables))
        elif metric == 'crossentropy':
            metrics.append(CrossentropyMetric(cfg, target_variables))
        else:
            raise ValueError(f'Unknown metric {cfg.metric}')

    return metrics


def lower_is_better(metric: str) -> bool:
    """Indicates whether lower values of the metric are better. """
    return metric not in ['nse', 'accuracy']
