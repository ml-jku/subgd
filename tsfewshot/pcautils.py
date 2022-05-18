import copy
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector

from tsfewshot import metrics
from tsfewshot.config import Config
from tsfewshot.models import get_model
from tsfewshot.utils import load_model, update_classification_head

LOGGER = logging.getLogger(__name__)


def get_best_epoch(run: Path) -> int:
    """Get best epoch of a run.

    Parameters
    ----------
    run : Path
        Run directory

    Returns
    -------
    int
        Index of the best epoch.
    """
    best_epoch_file = run / 'best_epoch.txt'
    if not best_epoch_file.exists():
        raise ValueError('No best epoch file in run: '+str(run))
    with best_epoch_file.open('r') as fp:
        return int(fp.read())


def load_diffs(cfg: Config,
               run: Path,
               init_vector: Union[torch.Tensor, Dict[str, torch.Tensor]],
               improved_epochs: bool = False,
               use_path: int = None,
               reduce_classification_head: int = None,
               use_steps: int = None,
               layerwise_diffs: bool = False) -> Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
    """Load vectors of weight differences.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run : Path
        Path to the run directory.
    init_vector : Union[torch.Tensor, Dict[str, torch.Tensor]]
        Initial parameter vector. If `layerwise_diffs` is True, this must be a dictionary of weights for each parameter.
    improved_epochs : bool, optional
        If True, will only consider epochs where the validation metric improved.
    use_path : int, optional
        If provided, will use every n-th epoch.
    reduce_classification_head : int, optional
        If provided, will average the head to the provided number of outputs.
    use_steps : int, optional
        If provided, will use every n-th gradient step.
    layerwise_diffs : bool, optional
        If True, this model will return a dict with parameter names as keys and lists of parameter
        differences as values.

    Returns
    -------
    Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]
        List of weight differences, either one global difference or differences along the training path.
        If `layerwise_diffs` is True, this will be a dictionary of weight difference for each parameter.
    """
    if (layerwise_diffs and not isinstance(init_vector, dict)) \
            or (layerwise_diffs and not isinstance(init_vector, dict)):
        raise ValueError('init_vector must be a dict when layerwise_diffs is true.')
    if layerwise_diffs:
        previous_vector: Dict[str, torch.Tensor] = \
            copy.deepcopy({name: param.reshape(-1) for name, param in init_vector.items()})  # type: ignore
    else:
        previous_vector = {'': copy.deepcopy(init_vector)}  # type: ignore
    best_epoch = get_best_epoch(run)

    if use_path is not None:
        epochs = list(range(use_path, best_epoch + 1, use_path))
        if best_epoch not in epochs:
            # make sure last step is included
            epochs.append(best_epoch)
    else:
        epochs = [best_epoch]

    if improved_epochs:
        keep_epochs = _filter_improved_epochs(run, epochs, cfg.metric[0])
        LOGGER.info(f'Keeping improved epochs {keep_epochs} (discarding {len(epochs) - len(keep_epochs)} epochs).')
        epochs = keep_epochs

    if use_steps is not None:
        max_step = max(int(model_name.name.split('step')[1][:-2]) for model_name in run.glob('model_epoch*_step*.p'))
        model_paths = [((epoch, step), run / f'model_epoch{str(epoch).zfill(3)}_step{step}.p')
                       for epoch, step in itertools.product(epochs, range(max_step + 1))]
        # make sure that we use values towards the end for short paths (::use_steps) would take the first step only
        model_paths = model_paths[::use_steps] if len(model_paths) > use_steps else model_paths[-1:]
    else:
        model_paths = [(epoch, run / f'model_epoch{str(epoch).zfill(3)}.p') for epoch in epochs]
    existing_model_paths = [(epoch, p) for (epoch, p) in model_paths if p.exists()]
    LOGGER.info(f'{run}: {best_epoch} epochs, subsample {[ep for ep, _ in existing_model_paths]}.')
    if len(model_paths) > len(existing_model_paths):
        LOGGER.warning(f'Skipping {len(model_paths) - len(existing_model_paths)} non-existing models.')
    deltas = {name: [] for name in previous_vector.keys()}
    for _, model_path in existing_model_paths:
        uninit_model = get_model(cfg, is_test=True, is_finetune=False)
        # average the head into a single set of weights, then replicate for the number of classes
        if reduce_classification_head is not None:
            uninit_model = update_classification_head(cfg, uninit_model)
        # if use_steps, we only have the parameters, not the states, so we load the model in non-strict mode
        current_model, _ = load_model(cfg, uninit_model, model_path=model_path, strict=not use_steps)

        if layerwise_diffs:
            current_vector = {name: param.reshape(-1) for name, param in current_model.named_parameters()}
        else:
            current_vector = {'': parameters_to_vector(current_model.parameters())}

        for name in deltas.keys():
            deltas[name].append((current_vector[name] - previous_vector[name]).detach().cpu())
        previous_vector = current_vector

    if not layerwise_diffs:
        deltas = deltas['']
    return deltas


def _filter_improved_epochs(run: Path, epochs: List[int], metric: str):
    previous_results_file = run / 'results' / f'test-model_epoch{str(epochs[0]).zfill(3)}-{metric}.csv'
    previous_results = pd.read_csv(previous_results_file, index_col=0)[metric].median()
    keep_epochs = [epochs[0]]
    for epoch in epochs[1:]:
        epoch_results_file = run / 'results' / f'test-model_epoch{str(epoch).zfill(3)}-{metric}.csv'
        epoch_results = pd.read_csv(epoch_results_file, index_col=0)[metric].median()
        if (metrics.lower_is_better(metric) and epoch_results < previous_results) \
                or (not metrics.lower_is_better(metric) and epoch_results > previous_results):
            keep_epochs.append(epoch)
            previous_results = epoch_results
    return keep_epochs
