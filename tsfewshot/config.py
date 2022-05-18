from __future__ import annotations

import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from ruamel.yaml import YAML

LOGGER = logging.getLogger(__name__)


class Config:
    """Configuration class for experiment runs.

    Parameters
    ----------
    cfg : Union[Path, str, dict]
        Dictionary or path to yaml file with configuration arguments.
    dev_mode : bool, optional
        If False, will only allow dict keys that correspond to a known config setting.
    """

    def __init__(self, cfg: Union[Path, str, dict], dev_mode: bool = False):
        if isinstance(cfg, (Path, str)):
            cfg_path = Path(cfg)
            if cfg_path.exists():
                with cfg_path.open('r') as fp:
                    yaml = YAML(typ="safe")
                    cfg = dict(yaml.load(fp))
            else:
                raise FileNotFoundError(cfg)
        self._cfg = cfg

        if not (self._cfg.get('dev_mode', False) or dev_mode):
            Config._check_cfg_keys(self._cfg)

    def update(self, other: Union[Config, dict], dev_mode: bool = False):
        """Update the configuration with settings from another configuration.

        Parameters
        ----------
        other : Union[Config, dict]
            The config to use for updating this config, either a dict or a Config instance.
        dev_mode : bool, optional
            If False, will only allow dict keys that correspond to a known config setting.
        """
        if not isinstance(other, Config):
            other = Config(other, dev_mode=dev_mode)  # convert to config to do dev_mode checks
        self._cfg.update(dict(other))

    # Training settings

    @property
    def base_dir(self) -> Path:
        """Base path of the dataset. """
        return Path(self._cfg['base_dir'])

    @property
    def base_run_dir(self) -> Path:
        """Base run directory for mode 'finetune'. """
        return Path(self._cfg['base_run_dir'])

    @property
    def batch_size(self) -> int:
        """Batch size for training. """
        return self._cfg['batch_size']

    @property
    def checkpoint_path(self) -> Path:
        """Model path to initialize the model on large-data finetuning (required for mode 'finetune'). """
        return Path(self._cfg['checkpoint_path'])

    @property
    def clip_gradient_norm(self) -> float:
        """Clip gradient norm to this value. """
        return self._cfg.get('clip_gradient_norm', None)

    @property
    def commit_hash(self) -> str:
        """Commit hash of the code directory. """
        return self._cfg.get('commit_hash', '')

    @commit_hash.setter
    def commit_hash(self, commit_hash: str):
        """Set the commit hash of the code directory. """
        self._cfg['commit_hash'] = commit_hash

    @property
    def dataset_max_size(self) -> int:
        """Limit the dataset size. Values <= 0 mean no restriction. """
        return self._cfg.get('dataset_max_size', -1)

    @property
    def dataset_max_trajectories(self) -> int:
        """Limit the number of trajectories per task. Values <= 0 mean no restriction.
        Only implemented for ThreeBody dataset. """
        if self.meta_dataset not in ['three_body', 'cstr', 'rlc']:
            raise NotImplementedError('Trajectory limit only implemented for ThreeBody and CSTR datasets.')
        return self._cfg.get('dataset_max_trajectories', -1)

    @property
    def filter_samples(self) -> Dict[str, Tuple[float, float]]:
        """Filter trajectories by their minimum/maximum value.
        Only implemented for Double Pendulum and ThreeBody datasets. """
        return self._cfg.get('filter_samples', {})

    @property
    def precalculated_scaler(self) -> Path:
        """Allows specifying a pre-calculated scaler instead of calculating it.
        Can be useful for performance purposes. """
        scaler_file = self._cfg.get('precalculated_scaler')
        if scaler_file is not None:
            scaler_file = Path(scaler_file)
        return scaler_file  # type: ignore

    @property
    def input_vars(self) -> Dict[str, List[str]]:
        """Input variables of the model.
        Optionally can be specified as a dict with keys 'train', 'finetune' to set different variables for
        training vs. finetuning/evaluation.
        """
        input_vars = self._cfg.get('input_vars', ['x'])
        if isinstance(input_vars, dict):
            if not sorted(input_vars.keys()) == ['finetune', 'train']:
                raise ValueError('If input_vars is a dict, it must have exactly the keys "train" and "finetune".')
        else:
            input_vars = {'train': input_vars, 'finetune': input_vars}  # same for training and finetuning/evaluation

        # make sure order is fixed by sorting alphabetically
        input_vars = {mode.lower(): sorted(input_list) for mode, input_list in input_vars.items()}
        if len(input_vars['train']) != len(input_vars['finetune']):
            raise ValueError('Must have same number of train and finetune input variables.')

        return input_vars

    @property
    def meta_dataset(self) -> str:
        """Name of the meta-dataset to use. """
        return self._cfg['meta_dataset'].lower()

    @property
    def output_size(self) -> int:
        """Number of target values, automatically calculated from target_vars if not a classification setting and from
        classification_n_classes if a classification setting. """
        # always use train to determine the number of classes. If it's different for finetune, the model loading code
        # will resize the head accordingly.
        output_size = self.classification_n_classes['train']
        if output_size is None:
            output_size = sum(len(v) for v in self.target_vars['train'].values())
        return output_size

    @property
    def classification_n_classes(self) -> Dict[str, int]:
        """Number of classes in classification settings for training and finetuning/evaluation.
        Either an int or a dict with keys 'train', 'finetune'. The latter allows for pretraining on all classes and
        subsequent finetuning in an N-way K-shot setting. """
        n_classes = self._cfg.get('classification_n_classes', None)
        if not isinstance(n_classes, dict):
            n_classes = {'train': n_classes, 'finetune': n_classes}
        else:
            if n_classes['train'] != n_classes['finetune'] and self.training_setup != 'supervised':
                raise ValueError('Different classes for train/finetune are only supported for supervised training.')
            if n_classes['train'] != n_classes['finetune'] \
                    and self.model not in ['lstm', 'feedforward', 'imagecnn', 'resnet']:
                raise NotImplementedError(f'Different classes for train/finetune are not supported for {self.model}.')
        return n_classes  # type: ignore

    @property
    def sort_episode_classes(self) -> bool:
        """If True, the classes within an episode will be sorted alphabetically. """
        return self._cfg.get('sort_episode_classes', True)

    @property
    def target_noise_std(self) -> float:
        """std of random noise to add to the target values during training/finetuning. """
        return self._cfg.get('target_noise_std', 0.0)

    @property
    def timeseries_is_sample(self) -> bool:
        """If True, one timeseries is considered to be one sample. If False, one target time step is one sample. """
        return self._cfg.get('timeseries_is_sample', False)

    @property
    def seq_length(self) -> int:
        """Length of the input sequences if `timeseries_is_sample` is False. """
        seq_length = self._cfg.get('seq_length', None)
        if seq_length is None and not self.timeseries_is_sample:
            raise ValueError('seq_length required if timeseries_is_sample is False')
        return seq_length  # type: ignore

    @property
    def target_vars(self) -> Dict[str, Dict[str, List[Union[int, str]]]]:
        """Prediction offsets or target time steps for each target.

        For each variable, one or multiple targets can be specified in a list. Integer list entries correspond to
        k-step-ahead prediction (positive k) or k-step-back reconstruction (negative k). Alternatively, one can
        specify a target as 'step_k', which corresponds to a constant target of the k-th time step.

        Optionally, one can specify different targets for training and finetuning (as dict with keys
        'train', 'finetune').
        """
        # make sure order is fixed by sorting alphabetically
        target_vars = self._cfg.get('target_vars', {var: [1] for var in self.input_vars['train']})
        if any(k.lower() in ['train', 'finetune'] for k in target_vars.keys()):
            if not self.finetune:
                LOGGER.warning('Specified different train and finetune targets, but finetune is False.')
            if not all(mode in [k.lower() for k in target_vars.keys()] for mode in ['train', 'finetune']):
                raise ValueError('If target_vars contains keys "train" or "finetune", it must contain both these keys.')
        else:
            # same targets for training and finetuning/evaluation
            target_vars = {'train': target_vars, 'finetune': target_vars}

        target_vars = {mode.lower(): {k: targets[k] for k in sorted(targets.keys())}  # type: ignore
                       for mode, targets in target_vars.items()}
        if len([target for target_list in target_vars['train'].values() for target in target_list]) \
                != len([target for target_list in target_vars['finetune'].values() for target in target_list]):
            raise ValueError('Must have same number of train and finetune targets.')

        return target_vars  # type: ignore

    @property
    def early_stopping_patience(self) -> int:
        """Number of epochs that don't achieve better validation scores before training is stopped. """
        return self._cfg.get('early_stopping_patience', None)

    @property
    def finetune_early_stopping_patience(self) -> int:
        """Number of epochs that don't achieve better validation scores before finetuning is stopped.
        Only relevant if finetune_eval_every is set. """
        return self._cfg.get('finetune_early_stopping_patience', None)

    @property
    def epochs(self) -> int:
        """Number of epochs to train for. """
        return self._cfg.get('epochs', 1)

    @property
    def eval_every(self) -> int:
        """Defines after how many epochs validation is run. """
        return self._cfg.get('eval_every', 1)

    @property
    def experiment_name(self) -> str:
        """Name of the experiment. """
        return self._cfg['experiment_name']

    @property
    def finetune(self) -> bool:
        """If True, the tester will finetune models on target tasks. """
        return self._cfg.get('finetune', False)

    @property
    def finetune_epochs(self) -> bool:
        """Number of epochs to finetune a PyTorch model (if cfg.finetune is True). """
        return self._cfg['finetune_epochs']

    @property
    def finetune_eval_every(self) -> Optional[Union[int, list]]:
        """If positive int, will evaluate the model during finetuning after every n updates. If list, will evaluate
        in the listed iterations. """
        return self._cfg.get('finetune_eval_every', None)

    @property
    def finetune_lr(self) -> Tuple[float, List[int], float]:
        """Learning rate for finetuning. Default: cfg.learning_rate.
        Either one float or Tuple (float, List[int], gamma) for MultiStepLR scheduler
        (initial learning rate, milestone epochs, reduction factor).
        """
        lr = self._cfg.get('finetune_lr', self.learning_rate)
        if isinstance(lr, float):
            lr = (lr, None, None)
        return lr  # type: ignore

    @property
    def finetune_setup(self) -> str:
        """Normal training or PCA. See test/__init__.py for all options. """
        return self._cfg.get('finetune_setup', '').lower()

    @property
    def finetune_regularization(self) -> List[Dict[str, Union[str, float, int]]]:
        """Regularization method to use during finetuning. List of dicts with at least keys 'method' and 'weight'. """
        regularizations = self._cfg.get('finetune_regularization', [])
        regularizations = [{k.lower(): v.lower() if isinstance(v, str) else v for k, v in reg.items()}
                           for reg in regularizations]
        if any('method' not in reg.keys() or 'weight' not in reg.keys() for reg in regularizations):
            raise ValueError('finetune_regularization entries must at least have keys "method" and "weight".')
        return regularizations  # type: ignore

    @property
    def learning_rate(self) -> Tuple[float, List[int], float]:
        """Learning rate for training. Either one float or Tuple (float, List[int], gamma) for MultiStepLR scheduler
        (initial learning rate, milestone epochs, reduction factor).
        """
        lr = self._cfg.get('learning_rate', 1e-3)
        if isinstance(lr, float):
            lr = (lr, None, None)
        return lr  # type: ignore

    @property
    def input_output_n_random_nets(self) -> int:
        """Number of input/output nets to randomly choose from during pretraining. Finetuning always uses only one net.

        This is intended to force the core model (between input and output layers) to learn more and different dynamics.
        Default: 1
        """
        return self._cfg.get('input_output_n_random_nets', 1)

    @property
    def input_output_types(self) -> Dict[str, str]:
        """Defines if actual values or deltas or both are used as inputs/outputs.
        Dict of style: {'input': 'values/deltas/both/image_vector', 'output': 'values/deltas/delta_t'}.
        For output, "deltas" means the difference between current and previous absolute target value, while
        "delta_t" means the difference between the current time step (at time of prediction)
        and the predicted time step.
        """
        input_output_types = self._cfg.get('input_output_types', {'input': 'values', 'output': 'values'})
        input_output_types = {k.lower(): v.lower() for k, v in input_output_types.items()}
        if input_output_types['input'] not in ['values', 'deltas', 'both'] \
                or input_output_types['output'] not in ['values', 'deltas', 'delta_t']:
            raise ValueError('input_output_types must be values/deltas/delta_t/both '
                             '(both only for input, delta_t only for output)')
        return input_output_types

    @property
    def ig_pca_file(self) -> Path:
        """Path to file on disk that stores PCA values to be used for finetuning. """
        file = Path(self._cfg['ig_pca_file'])
        if not file.is_file():
            raise FileNotFoundError(f'PCA file not found at {file}. Use pca.py to calculate PCA.')
        return file

    @property
    def use_pca_weights(self) -> bool:
        """If False, will not use PCA explained variance to weight the subspace in PCA-based finetuning. """
        return self._cfg.get('use_pca_weights', True)

    @property
    def pca_interpolation_factor(self) -> float:
        """Interpolation factor between Adam/SGD and PCA-based optimization. Only used if PCAOptimizer is used. """
        return self._cfg.get('pca_interpolation_factor', None)

    @property
    def pca_normalize(self) -> bool:
        """If True, will normalize PCA values to norm sqrt(number of parameters). """
        if 'integrated_gradients_normalize' in self._cfg.keys():
            return self.integrated_gradients_normalize
        return self._cfg.get('pca_normalize', True)

    @property
    def integrated_gradients_normalize(self) -> bool:
        """If True, will normalize PCA values to norm sqrt(number of parameters). """
        LOGGER.warning('integrated_gradients_normalize is deprecated. Use pca_normalize instead.')
        return self._cfg.get('integrated_gradients_normalize', True)

    @property
    def pca_sparsity(self) -> Tuple[str, float]:
        """Defines a sparsity for PCA. """
        spec = self._cfg.get('pca_sparsity', self.integrated_gradients_sparsity)
        if spec is not None:
            if len(spec) != 2:
                raise ValueError('PCA sparsity must be tuple (method, sparsity/threshold/quantile).')
            return (spec[0].lower(), spec[1])
        return None  # type: ignore

    @property
    def integrated_gradients_sparsity(self) -> Tuple[str, float]:
        """Deprecated. Use pca_sparsity. """
        return self._cfg.get('integrated_gradients_sparsity', None)

    @property
    def loss_fn(self) -> str:
        """Loss function to use for training. """
        return self._cfg.get('loss_fn', 'mse').lower()

    @property
    def maml_first_order(self) -> bool:
        """If True, will run MAML in first-order mode and ignore higher-order gradients. """
        return self._cfg.get('maml_first_order', False)

    @property
    def maml_n_inner_iter(self) -> int:
        """Number of inner loop iterations for MAML. """
        return self._cfg['maml_n_inner_iter']

    @property
    def maml_inner_lr(self) -> Union[Tuple[float, float], float]:
        """MAML/Reptile inner-loop learning rate or MetaSGD inner-loop learning rate range to sample from. """
        return self._cfg.get('maml_inner_lr', self.finetune_lr)

    @property
    def maml_inner_batch_size(self) -> int:
        """MAML/Reptile size of inner-loop batches. """
        inner_bs = self._cfg.get('maml_inner_batch_size', None)
        if inner_bs is None:
            inner_bs = self.support_size
        return inner_bs

    @property
    def metric(self) -> List[str]:
        """Metrics to use for validation and testing. Either a string or a list of strings.
        The first enty will be used as the early-stopping metric. """
        metrics = self._cfg.get('metric', 'RMSE')
        if isinstance(metrics, str):
            metrics = [metrics]
        return [m.lower() for m in metrics]

    @property
    def metric_aggregation(self) -> str:
        "How to select the best epoch (mean vs. median vs. global (i.e., metric calculated across all tasks)). "
        return self._cfg.get('metric_aggregation', 'mean').lower()

    @property
    def optimizer(self) -> Dict[str, str]:
        """Optimizer to use for training. Can be a dict with keys 'train', 'finetune', or just an optimizer name. """
        optimizer = self._cfg.get('optimizer', 'adam')
        if not isinstance(optimizer, dict):
            optimizer = {'train': optimizer, 'finetune': optimizer}
        if sorted(optimizer.keys()) != ['finetune', 'train']:
            raise ValueError('Must specify optizier as string or dict with keys "finetune", "train".')
        return {mode: optim.lower() for mode, optim in optimizer.items()}

    @property
    def piecewise_linear_ig(self) -> float:
        """If not None, will do piecewise linear approximation of the IG values.

        The number of iterations (``m``) will be calculated such that no step is larger than this value.
        Else, will only consider the gradient descent steps.
        """
        return self._cfg.get('piecewise_linear_ig', None)

    @property
    def plot_n_figures(self) -> int:
        """Number of datasets for which figures will be plotted during validation. """
        return self._cfg.get('plot_n_figures', None)

    @property
    def predict_last_n(self) -> int:
        """Number of steps on which the loss is computed. Earlier steps are ignored as warm-up. Default: all steps. """
        pred_last_n = self._cfg.get('predict_last_n', None)
        if pred_last_n is None and not self.timeseries_is_sample:
            raise ValueError('Must provide predict_last_n if timeseries_is_sample is False.')
        return pred_last_n  # type: ignore

    @property
    def query_size(self) -> int:
        """Size of each task's query set. """
        return self._cfg.get('query_size', 47)

    @property
    def run_dir(self) -> Optional[Path]:
        """Run directory of the experiment. """
        if 'run_dir' in self._cfg:
            return Path(self._cfg['run_dir'])
        return None

    @run_dir.setter
    def run_dir(self, run_dir: Path):
        self._cfg['run_dir'] = run_dir

    @property
    def save_every(self) -> int:
        """Defines how often the model is saved, in epochs.
        Negative values mean saving if the model improved in validation. """
        return self._cfg.get('save_every', 1)

    @property
    def save_predictions(self) -> bool:
        """If True, will save validation/test results to netcdf file. """
        return self._cfg.get('save_predictions', False)

    @property
    def store_training_path(self) -> bool:
        """If True, will keep track of the full training path and store it to disk.
        Only supported in supervised training/finetuning. """
        return self._cfg.get('store_training_path', False)

    @property
    def support_size(self) -> int:
        """Size of each task's support set. """
        return self._cfg.get('support_size', 3)

    @property
    def test_datasets(self) -> List[str]:
        """Datasets to use for testing. Can be specified as list of names of path to a file that contains them. """
        if isinstance(self._cfg['test_datasets'], list):
            return sorted(self._cfg['test_datasets'])
        return sorted(Config.read_datasets(self._cfg['test_datasets']))

    @property
    def train_datasets(self) -> List[str]:
        """Datasets to use for training. Can be specified as list of names of path to a file that contains them. """
        if isinstance(self._cfg['train_datasets'], list):
            return sorted(self._cfg['train_datasets'])
        return sorted(Config.read_datasets(self._cfg['train_datasets']))

    @property
    def train_val_split(self) -> float:
        """Option to create a supervised (non-meta-learning) train--validation split.
        If provided, this float will be the fraction of train samples to use during training; the remaining samples
        will be used in evaluation.
        Only supported if the train datasets are equal to the val datasets. """
        train_val_split = self._cfg.get('train_val_split', None)
        if train_val_split is not None and (train_val_split >= 1 or train_val_split <= 0):
            raise ValueError('train_val_split must be between 0 and 1.')
        return train_val_split  # type: ignore

    @property
    def dataset_subsets(self) -> Dict[str, Optional[Dict[str, Optional[List[int]]]]]:
        """List of indices to subset a dataset.
        Dict with keys in [train, val, test], where each value is a dict with one key per dataset in the corresponding
        split (train/val/test). The values of the inner dict are lists of indices to use in the subset.
        If a key is omitted, the corresponding dataset(s) will not be subset.
        """
        dataset_subsets = self._cfg.get('dataset_subsets', {'train': None, 'val': None, 'test': None})
        if any(key not in ['train', 'val', 'test'] for key in dataset_subsets.keys()):
            raise ValueError('dataset_subsets must be dict with keys in ["train", "val", "test"]')
        for missing_key in [k for k in ['train', 'val', 'test'] if k not in dataset_subsets.keys()]:
            dataset_subsets[missing_key] = None
        return dataset_subsets  # type: ignore

    @property
    def training_setup(self) -> str:
        """Indicates whether training works in supervised, maml, metasgd, reptile mode. """
        return self._cfg['training_setup'].lower()

    @property
    def val_datasets(self) -> List[str]:
        """Datasets to use for validation. Can be specified as list of names of path to a file that contains them. """
        if isinstance(self._cfg['val_datasets'], list):
            return sorted(self._cfg['val_datasets'])
        return sorted(Config.read_datasets(self._cfg['val_datasets']))

    @property
    def val_n_random_datasets(self) -> Optional[Union[List[Tuple[List[str], int]], int]]:
        """Number of randomly chosen datasets to evaluate during validation.
        None means all, zero or negative values mean no datasets.
        Can also directly provide a list of tasks, each task specified as follows: ``[[ds1, ds2, ...], seed]``.
        The list of datasets defines the n-way task, seed defines the random seed used to draw support and query set.
        """
        return self._cfg.get('val_n_random_datasets', None)

    @property
    def weight_decay(self) -> float:
        """Weight decay for training and finetuning.
        Note: this is in addition to what is specified in `finetune_regularization`. """
        return self._cfg.get('weight_decay', 0.0)

    @property
    def cnn_config(self) -> List[dict]:
        """Configuration of CNN layers.
        List of dicts with keys out_channels, kernel_size, activation, padding, stride, batch_norm, max_pool. """
        cnn_cfg = self._cfg['cnn_config']
        for i, layer in enumerate(cnn_cfg):
            if 'type' not in layer:
                cnn_cfg[i]['type'] = 'conv'
            if 'activation' not in layer:
                cnn_cfg[i]['activation'] = 'relu'
            if (layer['type'] == 'conv' and 'kernel_size' not in layer) or 'out_channels' not in layer:
                raise ValueError(f'CNN layer {i} missing out_channels or kernel_size')

        return cnn_cfg

    @property
    def eulerode_config(self) -> Dict:
        """Configuration of EulerODE model.
        Dict with keys input state_space and activation.
        state_space is a List[str] containing the names of the state variables.
        The observed state variables should have the same name as the target variables.
        These will be used to determine the initial condition in each batch during training.
        """
        eulerode_config = self._cfg.get('eulerode_config', None)
        return eulerode_config

    @property
    def resnet_config(self) -> Dict:
        """Configuration of Resnet. (WIP; so far only the type, rest is hard-coded.)"""
        resnet_config = self._cfg.get('resnet_config', None)
        return resnet_config

    @property
    def cnn_image_size(self) -> int:
        """Size of input images. Required for CNNs. """
        return self._cfg.get('cnn_image_size', None)

    @property
    def dropout(self) -> float:
        """Dropout to use during training. """
        return self._cfg.get('dropout', 0.0)

    @property
    def encoding_dim(self) -> int:
        """Number of positional encoding dimensions (relevant for encoding_type "cat"). """
        if self.encoding_type == 'cat':
            return self._cfg['encoding_dim']
        return len(self.input_vars['train'])

    @property
    def encoding_type(self) -> str:
        """Type of positional encoding to add to inputs ("sum" or "cat"). """
        return self._cfg.get('encoding_type', None)

    @property
    def finetune_modules(self) -> Optional[Union[List[str], Dict[str, str]]]:
        """Which model parts to be finetuned. None means finetune everything. """
        return self._cfg.get('finetune_modules', None)

    @property
    def hidden_size(self) -> int:
        """Hidden size of the model. """
        return self._cfg['hidden_size']

    @property
    def input_layer(self) -> dict:
        """Specification of the optional input layer. None/empty dict means no input layer """
        return self._cfg.get('input_layer', None)

    @property
    def layer_per_dataset(self) -> str:
        """If True, will use individual input/output weights for each dataset during training (not during finetuning).
        Only supported for supervised training.
        For input:
        Either 'singlerotation' (all timesteps of one input sample will use the same rotation), or 'mixedrotation'
        (all timesteps will use a random rotation).
        For output: 'output' (all rotations are predicted at once).
        """
        layer_per_dataset = self._cfg.get('layer_per_dataset', None)
        if layer_per_dataset and self.training_setup != 'supervised':
            raise ValueError('layer_per_dataset is only supported for supervised training.')
        if layer_per_dataset and self.model != 'lstm':
            raise NotImplementedError('Per-dataset input layer only supported for LSTM.')
        if layer_per_dataset and self.input_layer:
            raise NotImplementedError('Input-layer averaging is not implemented for input embedding layers.')
        if layer_per_dataset == 'output' \
                and (len(self.output_layer['hiddens']) > 1 or self.input_output_n_random_nets != 1):
            raise ValueError('Per-dataset output layer only supported for single-layer output.')
        if layer_per_dataset not in [None, 'singlerotation', 'mixedrotation', 'output']:
            raise ValueError(f'Unknown input strategy {layer_per_dataset}')
        return layer_per_dataset  # type: ignore

    @property
    def layer_per_dataset_eval(self) -> bool:
        """If True and layer_per_dataset is set, will also use the per-dataset weights in evaluation. """
        return self._cfg.get('layer_per_dataset_eval', False)

    @property
    def lstm_finetune_error_input(self) -> bool:
        """If True, the LSTM will take the previous time step's error as additional input on new tasks. """
        return self._cfg.get('lstm_finetune_error_input', False)

    @property
    def lstm_head_inputs(self) -> Dict[str, List[str]]:
        """LSTM outputs to use as inputs to the head during initial training and during finetuning.

        Can be: (a) list (options: h, c, i, f, g, o), then these values will be used for both training and finetuning;
        (b) dict with keys "train" and "finetune" where values are lists of str as in (a).
        Default: ['h'].
        Return type is always dict (options that match (a) are automatically transformed to their equivalent dict).
        """
        head_inputs = self._cfg.get('lstm_head_inputs', ['h'])
        if isinstance(head_inputs, list):
            head_inputs = {'train': head_inputs, 'finetune': head_inputs}
        if not isinstance(head_inputs, dict) or not sorted(head_inputs.keys()) == ['finetune', 'train']:
            raise ValueError(f'{head_inputs} must be list or dict with keys "train", "finetune"')
        return head_inputs

    @property
    def lstm_initial_forget_bias(self) -> int:
        """Initial forget gate bias of the LSTM. """
        return self._cfg.get('lstm_initial_forget_bias', None)

    @property
    def lstm_n_gates(self) -> Dict[str, Dict[str, int]]:
        """Number of LSTM gates during initial training and during finetuning.

        Can be: (a) int, then this value will be used for both training and finetuning for both forget and output gate;
        (b) dict with keys "train" and "finetune" where values are dicts with keys 'i', 'f', 'g', 'o' and int as values.
        Default: 1.
        Return type is always dict (options that match (a) are automatically transformed to their equivalent dict).
        """
        n_gates = self._cfg.get('lstm_n_gates', 1)
        if isinstance(n_gates, int):
            n_gates = {'i': n_gates, 'f': n_gates, 'g': n_gates, 'o': n_gates}
            n_gates = {'train': n_gates, 'finetune': n_gates}
        if not isinstance(n_gates, dict) or not sorted(n_gates.keys()) == ['finetune', 'train']:
            raise ValueError(f'{n_gates} must be int or dict with keys "train", "finetune"')
        if not all(sorted(vals.keys()) == ['f', 'g', 'i', 'o'] for vals in n_gates.values()):
            raise ValueError('lstm_n_gates dict must contains sub-dicts with keys "i", "f", "g", and "o".')
        return n_gates

    @property
    def lstm_num_layers(self) -> int:
        """Number of stacked LSTM layers. """
        return self._cfg.get('lstm_num_layers', 1)

    @property
    def model(self) -> str:
        """Model to use in the experiment. For options, see `tsfewshot.models.get_model`. """
        return self._cfg['model'].lower()

    @property
    def output_layer(self) -> dict:
        """Specification of the optional output layer. None/empty dict means no output layer. """
        default_spec = {'activation': None, 'dropout': 0.0, 'hiddens': [self.output_size], 'output_activation': None}
        return self._cfg.get('output_layer', default_spec)

    @property
    def batch_norm_mode(self) -> str:
        """Usage of batchnorm in evaluation mode.

        If 'conventional', batch norm layers will be in eval mode during evaluation.
        If 'transductive', batch norm layers will be in train mode during evaluation, i.e., use batch statistics
        from the query set rather than the running_mean/_var from training.
        If 'maml-conventional' (only for MAML training), batch norm layers will behave like 'conventional' in eval mode,
        but will also be set to eval mode for the MAML query set update during meta-training.
        If 'metabn', the running_mean/_var from the support set will be used for the query sets.
        For details, see http://proceedings.mlr.press/v119/bronskill20a/bronskill20a.pdf.
        """
        mode = self._cfg.get('batch_norm_mode', 'conventional')
        return mode.lower()

    @property
    def rnn_learn_initial_state(self) -> bool:
        """If True, h_0 (and c_0 for LSTM) will be learned parameters of recurrent models. """
        return self._cfg.get('rnn_learn_initial_state', False)

    @property
    def num_workers(self) -> int:
        """Number of parallel workers for data loading. """
        return self._cfg.get('num_workers', 4)

    @property
    def seed(self) -> int:
        """Random seed for training. """
        return self._cfg.get('seed', 0)

    @property
    def device(self) -> torch.device:
        """Device to run the model on. """
        return torch.device(self._cfg.get('device', 'cpu'))

    @device.setter
    def device(self, device: torch.device):
        """Set device property. """
        self._cfg['device'] = str(device)

    @property
    def log_tensorboard(self) -> bool:
        """Indicates whether tensorboard logging is active. """
        return self._cfg.get('log_tensorboard', True)

    @property
    def silent(self) -> bool:
        """If True, will not use progress bars and reduce logging. """
        return self._cfg.get('silent', False)

    @property
    def is_label_shared(self) -> bool:
        """Defines, whether the training and validation setting is label-shared."""
        return self._cfg.get('is_label_shared', True)

    def dump_config(self, yml_path: Path, overwrite: bool = False):
        """Dump the config to a yaml file.

        Parameters
        ----------
        file_path : Path
            Path where to dump the config yaml file
        overwrite : bool, optional
            If set to True, will allow overwriting an existing config file.

        Raises
        ------
        FileExistsError
            If a file with the specified name already exists and `overwrite` is False.
        """
        if not yml_path.exists() or overwrite:
            with yml_path.open('w') as fp:
                temp_cfg = {}
                for key, val in self._cfg.items():
                    if any(key.endswith(x) for x in ['_dir', '_path', '_file']) and key != 'store_training_path':
                        temp_cfg[key] = str(val)
                    else:
                        temp_cfg[key] = val

                yaml = YAML()
                yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
        else:
            raise FileExistsError(yml_path)

    def log_config(self):
        """Log the configuration values. """
        for k, v in self._cfg.items():
            LOGGER.info(f'{k}: {v}')

    @staticmethod
    def read_datasets(datasets_file: Union[Path, str]) -> List[str]:
        """Read list of dataset names from a file. """
        if isinstance(datasets_file, str):
            datasets_file = Path(datasets_file)
        with datasets_file.open('r') as fp:
            datasets = sorted(ds.strip() for ds in fp if ds.strip())
        return datasets

    @staticmethod
    def _check_cfg_keys(cfg: dict):
        """Checks the config for unknown keys. """
        property_names = [p for p in dir(Config) if isinstance(getattr(Config, p), property)]

        unknown_keys = [k for k in cfg.keys() if k not in property_names]
        if unknown_keys:
            raise ValueError(f'Unknown config keys: {unknown_keys}')

    def __iter__(self):
        """Iterate through a copy of the dict representation of the config. """
        return iter(self._cfg.copy().items())

    def __repr__(self):
        return self._cfg.__repr__()
