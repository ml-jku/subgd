import argparse
import copy
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, IncrementalPCA
from torch.nn.utils import parameters_to_vector

from tsfewshot.config import Config
from tsfewshot.models import get_model
from tsfewshot.pcautils import load_diffs
from tsfewshot.utils import load_model

logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO, format='%(asctime)s: %(message)s')


def exception_logging(type, value, tb):
    """Make sure we log uncaught exceptions. """
    LOGGER.exception('Uncaught exception', exc_info=(type, value, tb))


sys.excepthook = exception_logging

LOGGER = logging.getLogger(__name__)

np.random.seed(0)
torch.manual_seed(0)


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, help='Pretraining run directory')
    parser.add_argument('--finetune-dirs', type=str, nargs='+', help='Directories of finetuned model runs')
    parser.add_argument('--epoch', type=int, required=False, help='Epoch from pretrained model to use as base')

    parser.add_argument('--n-runs', required=False, type=int, help='Number of runs to use for PCA calculation')
    parser.add_argument('--n-components', required=False, type=int, help='Number of components to keep')
    parser.add_argument('--typ', default='torch', type=str, help='Method to calculate PCA')
    parser.add_argument('--batch-size', default=10000, type=int, help='Batch size for incremental PCA calculation')
    parser.add_argument('--centering', action='store_true', help='Whether the PCA is centered or not')
    parser.add_argument('--use-path', required=False, type=int, help='Use every n-th epoch')
    parser.add_argument('--use-steps', required=False, type=int, help='Use every n-th update step within an epoch')
    parser.add_argument('--device', default='cpu', type=str, help='device to calculate PCA on')
    parser.add_argument('--reduce-classification-head', required=False, type=int,
                        help='Number of classes during finetuning, if pretraining used more classes.')
    parser.add_argument('--improved-epochs', action='store_true',
                        help='Whether only epochs with improved val metric are used.')
    parser.add_argument('--layerwise-pca', action='store_true', help='Whether one PCA per parameter is calculated.')

    args = vars(parser.parse_args())
    args['__timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if len(args['finetune_dirs']) == 0:
        raise ValueError('Need finetune-dirs to calculate PCA')

    return args


def run_pca(x, typ: str, n=None):
    """
    returns a matrix to reduce the dimension of the input
    to the given number of dimensions. A matrix to 'restore'
    the reduced data back to its original form also returned
    """
    # https://medium.com/@rcorbish/principal-components-analysis-pca-f33113b8b05e
    cov = x.t().mm(x) / (x.shape[0] - 1)
    if typ == 'custom-numpy':
        x_np = x.numpy()
        cov = np.matmul(x_np.transpose(), x_np) / (x_np.shape[0] - 1)
        u, s, v = np.linalg.svd(cov)
        v = v.transpose()
    elif typ in ['custom', 'custom-torch']:
        cov = x.t().mm(x) / (x.shape[0] - 1)
        u, s, v = torch.svd(cov)
    else:
        raise ValueError('Unknown type')
    if n is not None:
        u1 = u[:, 0:n]
        v1 = v[0:n, :]
    else:
        u1 = u
        v1 = v
    if typ == 'custom-numpy':
        u1 = torch.from_numpy(u1)
        s = torch.from_numpy(s)
        v1 = torch.from_numpy(v1)

    return x.mm(u1), s, v1


if __name__ == '__main__':
    args = _get_args()
    base_dir = Path(args['base_dir'])
    runs = sorted(Path(r) for r in args['finetune_dirs'])

    LOGGER.info(args)
    LOGGER.info(f'{len(runs)} runs')

    n_runs = args["n_runs"]
    pca_path = (base_dir / f'pca/pca-{args["typ"]}{f"{n_runs}runs" if n_runs else ""}'
                f'center{args["centering"]}{args["n_components"]}componentspath{args["use_path"]}'
                f'improved{args["improved_epochs"]}usesteps{args["use_steps"]}layerwise{args["layerwise_pca"]}.p')
    if pca_path.exists():
        raise ValueError(f'PCA file exists at {pca_path}.')

    if n_runs:
        runs = np.random.choice(runs, size=n_runs, replace=False).tolist()

    init_epoch = args['epoch']
    if init_epoch is None:
        init_epoch = -1

    cfg = Config(base_dir / 'config.yml')
    cfg.update({'layer_per_dataset_eval': False, 'device': 'cpu'})
    reduce_classification_head = args["reduce_classification_head"]
    if reduce_classification_head is not None:
        cfg.update({'classification_n_classes': {'train': cfg.classification_n_classes['train'],
                                                 'finetune': reduce_classification_head}})

    uninit_model = get_model(cfg, is_test=True, is_finetune=False)
    init_model, _ = load_model(cfg, uninit_model, epoch=init_epoch)
    print({k: v.shape for k, v in init_model.named_parameters()})
    layerwise_pca = args["layerwise_pca"]
    if layerwise_pca:
        init_vector = {name: param.detach().cpu() for name, param in init_model.named_parameters()}
    else:
        init_vector = parameters_to_vector(init_model.parameters()).detach().cpu()

    cfg.update({'layer_per_dataset': None, 'device': 'cpu'})
    deltas = Parallel(n_jobs=1, verbose=1)(delayed(load_diffs)(cfg,
                                                               run,
                                                               copy.deepcopy(init_vector),
                                                               args["improved_epochs"],
                                                               args["use_path"],
                                                               reduce_classification_head,
                                                               args["use_steps"],
                                                               layerwise_pca) for run in runs)

    device = args["device"]
    if layerwise_pca:
        deltas = {name: torch.stack([vec for run_deltas in deltas  # type: ignore
                                     for vec in run_deltas[name]]).to(device)
                  for name in deltas[0].keys()}  # type: ignore
    else:
        deltas = {'': torch.stack([vec for run_deltas in deltas for vec in run_deltas]).to(device)}  # type: ignore

    param_pcas = {}
    typ = args["typ"]
    n_components = args["n_components"]
    for param_name, param_deltas in deltas.items():
        LOGGER.info(f'Fitting {typ} PCA on vector of shape {param_deltas.shape} '
                    f'(parameter {param_name if param_name != "" else "all"})')

        if typ == 'sklearn':
            pca = PCA(random_state=0)
            transformed_vector = pca.fit_transform(param_deltas)
            var_preserved = pca.explained_variance_ratio_
            pca_v = torch.from_numpy(pca.components_).t()
        elif typ == 'inc':
            pca = IncrementalPCA(batch_size=args["batch_size"], n_components=n_components)
            transformed_vector = pca.fit_transform(param_deltas)
            var_preserved = pca.explained_variance_ratio_
            pca_v = torch.from_numpy(pca.components_).t()
        elif typ == 'torch':
            mean = param_deltas.mean(dim=0)
            pca_u, pca_s, pca_v = torch.pca_lowrank(param_deltas,
                                                    q=min(*param_deltas.shape) if n_components is None
                                                    else n_components,
                                                    center=args["centering"])
            var_preserved = pca_s / pca_s.sum()
            pca = {'u': pca_u.cpu(), 's': pca_s.cpu(), 'v': pca_v.cpu(), 'explained_variance': var_preserved.cpu(),
                   'deltas_mean': mean.cpu()}
        elif typ in ['custom', 'custom-numpy', 'custom-torch']:
            transformed_vector, pca_s, pca_v = run_pca(param_deltas, typ, n=None)
            transformed_vector = transformed_vector.numpy()
            var_preserved = (pca_s / pca_s.sum()).numpy()
            pca = {'u': None, 'v': pca_v, 'explained_variance': var_preserved}
        else:
            raise ValueError('Unknown calculation type')

        LOGGER.info(f'Explained variance: {pca["explained_variance"].detach().numpy()}')  # type: ignore
        param_pcas[param_name] = pca

    if not layerwise_pca:
        param_pcas = param_pcas['']

    param_pcas['__args'] = args
    (base_dir / 'pca').mkdir(exist_ok=True)
    pickle.dump(param_pcas, pca_path.open('wb'))
    LOGGER.info(f'PCA saved at {pca_path}.')
