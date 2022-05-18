import copy
import hashlib
import itertools
import logging
import os
import pickle
import random
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn, optim

from tsfewshot.config import Config
from tsfewshot.optim.pcaoptimizer import PCAOptimizer

LOGGER = logging.getLogger(__name__)


def setup_run_dir(cfg: Config) -> Config:
    """Create run directory with unique name. """
    now = datetime.now()
    year = f'{now.year}'[2:]
    month = f'{now.month}'.zfill(2)
    day = f'{now.day}'.zfill(2)
    hour = f'{now.hour}'.zfill(2)
    minute = f'{now.minute}'.zfill(2)
    second = f'{now.second}'.zfill(2)
    run_name = f'{cfg.experiment_name}_{year}{month}{day}_{hour}{minute}{second}'

    if cfg.run_dir is None:
        run_dir = Path().cwd() / 'runs' / run_name
    else:
        run_dir = cfg.run_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=False)

    cfg.run_dir = run_dir

    return cfg


def setup_logging(log_file: str):
    """Initialize logging to `log_file` and stdout.

    Parameters
    ----------
    log_file : str
        Name of the log file.
    """
    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(handlers=[file_handler, stdout_handler], level=logging.INFO, format='%(asctime)s: %(message)s')

    # Log uncaught exceptions
    def exception_logging(typ, value, traceback):
        LOGGER.exception('Uncaught exception', exc_info=(typ, value, traceback))

    sys.excepthook = exception_logging

    LOGGER.info(f'Logging to {log_file} initialized.')


def get_random_states():
    """Return current random seeds to restore them later. """
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def set_seed(seed: Union[int, tuple]):
    """Set random seed or restore state as fetched by `get_random_states`. """
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        random.setstate(seed[0])
        np.random.set_state(seed[1])
        torch.random.set_rng_state(seed[2])


def save_model(model: 'BaseModel', model_path: Path):  # type: ignore
    """Save model to disk.

    Parameters
    ----------
    model : BaseModel
        Model to save.
    model_path : Union[Path, str]
        Path to the file where the model will be stored.
    """
    if model.is_pytorch:
        torch.save(model.state_dict(), str(model_path))
    else:
        with model_path.open('wb') as fp:
            pickle.dump(model, fp)


def get_activation(name: str) -> nn.Module:
    """Return an activation function of the specified name.

    Parameters
    ----------
    name : {'tanh', 'sigmoid', 'linear', 'relu', 'leakyreluX', 'softmax'}
        Name of the activation function, where X is a float that describes the negative LeakyReLU slope.

    Returns
    -------
    nn.Module
        Activation function instance.

    Raises
    ------
    NotImplementedError
        If the specified name does not correspond to a known activation function.
    """
    if name.lower() == 'tanh':
        activation = nn.Tanh()
    elif name.lower() == 'sigmoid':
        activation = nn.Sigmoid()
    elif name.lower() == 'linear':
        activation = nn.Identity()
    elif name.lower() == 'relu':
        activation = nn.ReLU()
    elif name.lower().startswith('leakyrelu'):
        slope = float(name.replace('leakyrelu', ''))
        activation = nn.LeakyReLU(slope)
    elif name.lower() == 'softmax':
        activation = nn.Softmax(dim=2)
    else:
        raise NotImplementedError(f'Activation {name} not supported')
    return activation


def get_optimizer_and_scheduler(name: str,
                                model: nn.Module,
                                learning_rate: float,
                                milestones: List[int] = None,
                                gamma: float = None,
                                weight_decay: float = 0.0,
                                cfg: Config = None,
                                pca: dict = None) \
        -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler.MultiStepLR]]:
    """Get optimizer and optionally a learning rate scheduler for the passed model.

    Parameters
    ----------
    name : str
        Name of the optimizer.
    model : nn.Module
        Model to optimize.
    learning_rate : float
        Initial learning rate of the optimizer.
    milestones : List[int], optional
        If provided, will also return a MultiStepLR learning rate scheduler.
    gamma : float, optional
        Reduction factor of the LR scheduler.
    weight_decay : float, optional
        Weight decay.
    cfg : Config
        Run configuration. Required for PCA optimizer.
    pca : dict, optional
        PCA loaded from disk. If provided, will use PCA together with sgd or adam.

    Returns
    -------
    Tuple[optim.Optimizer, Optional[optim.lr_scheduler.MultiStepLR]]
        Optimizer for the passed model and possibly a learning rate scheduler.
    """
    if pca is not None:
        if cfg is None:
            raise ValueError('Config is required for PCAOptimizer.')
        if weight_decay != 0:
            raise NotImplementedError('Weight decay not implemented for PCA.')
        optimizer = PCAOptimizer(cfg, name.lower(), dict(model.named_parameters()), learning_rate, pca)
    elif name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    elif name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer {name}.')

    scheduler = None
    if milestones:
        if gamma is None:
            raise ValueError('Need to provide milestones and gamma together.')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
    return optimizer, scheduler


def create_config_files(base_config_path: Union[str, Path], modify_dict: Dict[str, list], output_dir: Union[str, Path]):
    """Create variations of a config given a base config and a dictionary of modification lists.

    Parameters
    ----------
    base_config_path : Union[str, Path]
        Path to a base config file (.yml)
    modify_dict : dict
        Dictionary mapping from parameter names to lists of possible parameter values.
    output_dir : Union[str, Path]
        Path to a folder where the generated configs will be stored.
    """
    if isinstance(base_config_path, str):
        base_config_path = Path(base_config_path)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # load base config as dictionary
    base_config = Config(base_config_path)
    experiment_name = base_config.experiment_name
    option_names = modify_dict.keys()

    # iterate over each possible combination of hyper parameters
    for i, options in enumerate(itertools.product(*modify_dict.values())):

        base_config.update(dict(zip(option_names, options)))

        # create a unique run name
        name = experiment_name
        for key, val in zip(option_names, options):
            name += f'_{key}{val}'
        base_config.update({'experiment_name': name})

        base_config.dump_config(output_dir / f'config_{i+1}.yml')

    print(f'Done. Configs stored in {output_dir}')


def load_model(cfg: Config,
               uninit_model: 'BaseModel' = None,  # type: ignore
               epoch: int = None,
               model_path: Path = None,
               strict: bool = True) -> Tuple['BaseModel', str]:  # type: ignore
    """Load model from disk.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    uninit_model : BaseModel, optional
        Uninitialized model into which the weights will be loaded. Required for PyTorch models.
    epoch : int, optional
        Epoch for which the model will be loaded. If -1, will try to use the epoch specified in
        run_dir/best_epoch.txt. Default: last epoch stored to disk. Mutually exclusive with `model_path`.
    model_path : Path, optional
        Path to the model to be loaded. Mutually exclusive with `epoch`.
    strict : bool, optional
        If False, will use non-strict mode to load PyTorch models

    Returns
    -------
    Tuple[BaseModel, str]
        (Model loaded from disk, name of the model file)

    Raises
    ------
    FileNotFoundError
        If no model exists on disk for epoch `epoch`, or (if ``epoch is None``) if no model exists on disk.
    """
    if not strict:
        LOGGER.warning('Loading model in non-strict mode')
    if model_path is None:
        if epoch is None:
            model_paths = list(cfg.run_dir.glob('model_epoch*.p'))
            if len(model_paths) == 0:
                raise FileNotFoundError(f'No model found at {cfg.run_dir}')
            # we only do zfill to 3 digits, so for experiments with >1K epochs we need to sort numerically.
            model_epochs = np.array([int(name.stem[11:-2]) for name in model_paths])
            model_path = model_paths[np.argmax(model_epochs)]
        else:
            if epoch == -1:
                best_epoch_file = cfg.run_dir / 'best_epoch.txt'
                if not best_epoch_file.exists():
                    raise FileNotFoundError(f'best-epoch file {best_epoch_file} not found.')
                with best_epoch_file.open('r') as fp:
                    epoch = int(fp.read())
                LOGGER.info(f'Using best model (epoch {epoch})')

            model_path = cfg.run_dir / f'model_epoch{str(epoch).zfill(3)}.p'
    else:
        if epoch is not None:
            raise ValueError('Cannot provide model_path and epochs.')
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found at {model_path}')

    if uninit_model is not None and uninit_model.is_pytorch:
        if cfg.classification_n_classes['train'] != cfg.classification_n_classes['finetune'] \
                and cfg.layer_per_dataset is not None:
            raise ValueError('Per-dataset layer and changing class numbers are not supported together.')

        # this is somewhat hacky: We've saved the model with a large layer (one set of weights per dataset).
        # uninit_model already has the normal, small layer that will be used for finetuning (unless
        # cfg.layer_per_dataset_eval is True).
        # So we first create a large layer for it, load the weights, and then average the per-dataset weights into
        # the small original layer if requested.
        if cfg.layer_per_dataset in ['mixedrotation', 'singlerotation']:
            uninit_model.lstm.input_size = uninit_model.lstm.input_size * len(cfg.train_datasets)
            uninit_model.lstm.weight_ih_l0 = \
                nn.Parameter(torch.zeros((uninit_model.lstm.weight_ih_l0.shape[0],
                                          len(cfg.train_datasets)
                                          * uninit_model.input_layer.output_size)).to(uninit_model.device))
            uninit_model.load_state_dict(torch.load(str(model_path), map_location=cfg.device), strict=strict)

            if not cfg.layer_per_dataset_eval:
                uninit_model = reduce_per_dataset_layer(cfg, uninit_model)
        elif cfg.layer_per_dataset == 'output':
            head = uninit_model.head.networks[0].fc[0]
            head.out_features = head.out_features * len(cfg.train_datasets)
            head.weight = \
                nn.Parameter(torch.zeros((head.weight.shape[0] * len(cfg.train_datasets),
                                          head.weight.shape[1])).to(uninit_model.device))
            head.bias = \
                nn.Parameter(torch.zeros((len(cfg.train_datasets)
                                          * head.bias.shape[0])).to(uninit_model.device))
            uninit_model.load_state_dict(torch.load(str(model_path), map_location=cfg.device), strict=strict)

            if not cfg.layer_per_dataset_eval:
                uninit_model = reduce_per_dataset_layer(cfg, uninit_model)

        else:
            uninit_model.load_state_dict(torch.load(str(model_path), map_location=cfg.device), strict=strict)

            # average the head into a single set of weights, then replicate for the number of classes
            if cfg.classification_n_classes['train'] != cfg.classification_n_classes['finetune']:
                uninit_model = update_classification_head(cfg, uninit_model)

    else:
        with model_path.open('rb') as fp:
            uninit_model = pickle.load(fp)

    return uninit_model, model_path.stem


def update_classification_head(cfg: Config, model: 'BaseModel') -> 'BaseModel':  # type: ignore
    """Update a classification head from training for the number of finetuning/evaluation classes.

    This method first reduces the weights to all training outputs into one averaged weight vector and then repeats
    this vector to the number of finetuning/evaluation classes.

    Parameters
    ----------
    cfg : Config
        The run configuration
    model : BasePytorchModel
        The model to be updated

    Returns
    -------
    BasePytorchModel
        The model with a head of size ``cfg.classification_n_classes['finetune']``.
    """
    model = copy.deepcopy(model)
    head = model.head.networks[0].fc[-1]
    if cfg.classification_n_classes['train'] != cfg.classification_n_classes['finetune'] \
            and head.out_features != cfg.classification_n_classes['finetune']:
        head.out_features = cfg.classification_n_classes['finetune']
        head.weight = nn.Parameter(torch.mean(head.weight, dim=0, keepdim=True)  # type: ignore
                                   .repeat(head.out_features, 1).to(model.device))
        head.bias = nn.Parameter(torch.mean(head.bias, dim=0, keepdim=True)  # type: ignore
                                 .repeat(head.out_features).to(model.device))
        if hasattr(model, 'lstm'):
            model.lstm.flatten_parameters()
    return model


def reduce_per_dataset_layer(cfg: Config, model: 'LSTM', choose: int = None) -> 'LSTM':  # type: ignore
    """Reduce an LSTM's input/output layer with per-dataset weights into a normal layer that is shared across datasets,
    either by averaging the per-dataset weights, or by choosing one set of weights.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    model : LSTM
        The model with the large layer to be averaged.
    choose : int, optional
        If provided, will use the chosen set of weights instead of averaging all sets.

    Returns
    -------
    LSTM
        The passed model with the averaged layer.
    """
    model = copy.deepcopy(model)

    # note that in the following cases the order of reshaped dimensions is important!
    # we want to average the weights that belong to the different rotations, not to the different in-/output variables.
    if cfg.layer_per_dataset in ['mixedrotation', 'singlerotation']:

        if model.lstm.input_size == model.input_layer.output_size:
            # if we're in mode 'finetune', we already train the reduced model, so we don't need to do anything
            model.lstm.flatten_parameters()
            return model

        # reshape to (4*hidden, #datasets, input size). Input size is the output size
        # of the identity input layer (non-identity input layers are not allowed, see config.py).
        model.lstm.input_size = model.input_layer.output_size
        input_weights = model.lstm.weight_ih_l0
        input_weights = input_weights.reshape(input_weights.shape[0],
                                              len(cfg.train_datasets),
                                              model.input_layer.output_size)
        if choose is not None:
            model.lstm.weight_ih_l0 = input_weights[:, choose].contiguous()
        else:
            model.lstm.weight_ih_l0 = nn.Parameter(torch.mean(input_weights, dim=1))
    elif cfg.layer_per_dataset == 'output':
        head = model.head.networks[0].fc[0]
        if head.out_features == model.head.output_size:
            # if we're in mode 'finetune', we already train the reduced model, so we don't need to do anything
            model.lstm.flatten_parameters()
            return model

        head.out_features = model.head.output_size
        head_weights = head.weight.reshape(len(cfg.train_datasets),
                                           model.head.output_size,
                                           head.weight.shape[1])
        head_bias = head.bias.reshape(len(cfg.train_datasets), model.head.output_size)
        if choose is not None:
            head.weight = nn.Parameter(head_weights[choose].contiguous())
            head.bias = nn.Parameter(head_bias[choose].contiguous())
        else:
            head.weight = nn.Parameter(torch.mean(head_weights, dim=0))
            head.bias = nn.Parameter(torch.mean(head_bias, dim=0))
    else:
        raise ValueError('Unknown layer_per_dataset strategy')
    model.lstm.flatten_parameters()

    return model


def get_git_hash() -> Optional[str]:
    """Try to get the git hash of tsfewshot.

    Returns
    -------
    Optional[str]
        Git hash if git is installed and the project is cloned via git, None otherwise.
    """
    current_dir = str(Path(__file__).absolute().parent)
    try:
        if subprocess.call(['git', '-C', current_dir, 'branch'], stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL) == 0:
            git_output = subprocess.check_output(['git', '-C', current_dir, 'describe', '--always'])
            return git_output.strip().decode('ascii')
    except OSError:
        pass  # git is probably not installed
    return None


def save_git_diff(run_dir: Path):
    """Try to store the git diff of tsfewshot to a zip file.

    Parameters
    ----------
    run_dir : Path
        Directory of the current run
    """
    current_dir = str(Path(__file__).absolute().parent)
    try:
        out = subprocess.check_output(['git', '-C', current_dir, 'diff'], stderr=subprocess.DEVNULL)
    except OSError:
        return
    new_diff = out.strip().decode('utf-8')

    existing_diffs = list(run_dir.glob('git_diff*.zip'))
    file_path = run_dir / f'git_diff-{len(existing_diffs)}.zip'
    if len(existing_diffs) > 0:
        previous_zipfile = run_dir / f'git_diff-{len(existing_diffs) - 1}.zip'
        with zipfile.ZipFile(previous_zipfile, 'r') as last_zip:
            try:
                last_diff = last_zip.read('tsfewshot_diff.diff').decode('utf-8')
            except KeyError:
                LOGGER.warning(f'Could not read diff file from {previous_zipfile}.')
                last_diff = None
        if last_diff != new_diff:
            LOGGER.warning(f'Git diff changed. Writing new diff to {file_path}.')
        else:
            LOGGER.info(f'Diff unchanged from {previous_zipfile}.')
            return

    zip_file = zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED)  # type: ignore
    zip_file.writestr('tsfewshot_diff.diff', new_diff)
    zip_file.close()


def get_file_path(save_dir: Path, split: str, ds_name: str, epoch: str, ext: str) -> Path:
    """Construct a path to a file that contains split, dataset name, and epoch in its name.

    This method can cope with file names that are too long for the file system by shortening the path
    and storing a mapping from short to original name to disk.

    Parameters
    ----------
    save_dir : Path
        Directory where the file should be created.
    split : str
        Dataset split, will be part of the file name.
    ds_name : str
        Dataset name, will be part of the file name.
    epoch : str
        Epoch, will be part of the file name.
    ext : str
        File extension.

    Returns
    -------
    Path
        Constructed file path.
    """

    file_name = re.sub(r'[^A-Za-z0-9\._\-]+', '', f'{split}_{ds_name}_epoch{epoch}.{ext}')

    max_length = os.statvfs(save_dir).f_namemax
    if len(file_name) > max_length:
        name_map_file = save_dir / 'file_names.txt'
        ds_hash = hashlib.md5(file_name.encode()).hexdigest()
        short_name = f'{split}_{ds_hash}_epoch{epoch}.{ext}'
        LOGGER.warning(f'File name too long. Shortening {file_name} to {short_name}, '
                       f'storing mapping in {name_map_file.name}.')
        with name_map_file.open('a') as f:
            f.write(f'{file_name} -> {short_name}\n')
        file_name = short_name

    return save_dir / file_name


def proj_func_pos(s: torch.Tensor, l1_norm: float, l2_norm: float) -> torch.Tensor:
    """Find the closest non-negative vector with a given L1 and L2 norm.

    Ported to Python from this R function:
    https://www.rdocumentation.org/packages/fabia/versions/2.18.0/topics/projFuncPos

    Originally proposed in [#]_.

    Parameters
    ----------
    s : torch.Tensor
        Vector to be projected.
    l1 : float
        L1 norm of the projected vector.
    l2 : float
        L2 norm of the projected vector.

    Returns
    -------
    torch.Tensor
        The non-negative sparse projected vector.

    References
    ----------
    .. [#] Patrik O. Hoyer, "Non-negative Matrix Factorization with Sparseness Constraints",
           Journal of Machine Learning Research 5:1457-1469, 2004.
    """
    n = len(s)
    ones = torch.ones(n).to(s.device)

    # Start by projecting the point to the sum constraint hyperplane
    v = s + (l1_norm - torch.sum(s)) / n

    # Initialize zerocoeff (initially, no elements are assumed zero)
    zerocoeff = torch.zeros(len(v)).to(torch.bool).to(s.device)

    j = 0
    end = 0
    while end < 1:
        # This does the proposed projection operator
        midpoint = ones * l1_norm / (n - torch.sum(zerocoeff))
        midpoint[zerocoeff] = 0
        w = v - midpoint
        a = torch.sum(torch.square(w))
        b = 2 * torch.dot(w, v)
        c = torch.sum(torch.square(v)) - l2_norm
        t = torch.square(b) - 4 * a * c
        if t < 0:
            t = torch.tensor(0.0, device=s.device)  # pylint: disable=not-callable
        if a < 1e-10:
            a = torch.tensor(1e-10, device=s.device)  # pylint: disable=not-callable

        alphap = (-b + torch.sqrt(t)) / (2 * a)
        v = alphap * w + v

        if ((v >= 0).all()) and (j > 1):
            end = 2
        else:
            j += 1

            # Set negs to zero, subtract appropriate amount from rest
            zerocoeff = (v <= 0)
            v[zerocoeff] = 0
            tempsum = torch.sum(v)
            tta = n - torch.sum(zerocoeff)
            if tta > 0:
                v = v + (l1_norm - tempsum) / tta

            v[zerocoeff] = 0

    return v
