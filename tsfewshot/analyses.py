import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

from tsfewshot import metrics
from tsfewshot.config import Config
from tsfewshot.test import get_tester

LOGGER = logging.getLogger(__name__)


def get_combination_name(combination, i_seed):
    support = combination['support']
    lr = combination['lr']
    type_spec = combination['type']
    noise = combination['noise']
    sps = combination['sparsity']
    wd = combination.get('weight_decay')
    pca_interpolate = combination.get('pca_interpolate')
    sparse_str = f'{sps[0]}{sps[1]}' if sps is not None else 'None'
    name = f'support{support}_lr{lr}_{type_spec}_sparse{sparse_str}_noise{noise}_seed{i_seed}'
    if wd is not None:
        name += f'_weightdecay{wd}'
    if pca_interpolate is not None:
        name += f'_pcainterpolate{pca_interpolate}'
    return name.replace('/', '-').replace(' ', '-').replace('(', '_').replace(')', '_')


def create_finetune_configs(runs, tasks, lr,
                            epoch: int = None,
                            eval_every: int = 1,
                            save_every: int = 1,
                            additional_args: str = None,
                            suffix: str = ''):
    for run in runs:
        finetune_dir = f'finetune_pca_epoch{str(epoch) if epoch is not None else ""}_lr{lr}'
        if epoch is None:
            best_epoch_file = run / 'best_epoch.txt'
            if not best_epoch_file.exists():
                print(f'best-epoch file {best_epoch_file} not found.')
                continue
            with best_epoch_file.open('r') as fp:
                epoch = int(fp.read())

        finetune_cfg = f"""
experiment_name: NNNN
test_datasets: TTTT
val_datasets: TTTT
finetune_lr: {lr}
finetune_epochs: 1000
optimizer: adam
early_stopping_patience: 20
eval_every: {eval_every}
save_every: {save_every}
checkpoint_path: {str(run.absolute())}/model_epoch{str(epoch).zfill(3)}.p
base_run_dir: {str(run.absolute())}
run_dir: {str(run.absolute())}/{finetune_dir}
num_workers: 4
"""
        if additional_args is not None:
            finetune_cfg += additional_args

        (run / f'{finetune_dir}/configs').mkdir(exist_ok=True, parents=True)
        for task in tasks:
            experiment_name = f'{run.name}-{"".join(task)}{suffix}'
            # we'll append a timestamp to the name when creating the rundir, so keep 14 chars free
            if len(experiment_name) >= os.statvfs(run).f_namemax - 14:
                short_experiment_name = f'{run.name}-{hashlib.md5("".join(task).encode()).hexdigest()}{suffix}'
                LOGGER.warning(f'Shortening experiment name from {experiment_name} to {short_experiment_name}.')
                experiment_name = short_experiment_name
            task_finetune_cfg = finetune_cfg.replace('TTTT', str(task)).replace('NNNN', experiment_name)
            file_name = f'{"".join(task).replace("#", "-")}{suffix}.yml'
            if len(file_name) > os.statvfs(run).f_namemax:
                short_file_name = f'{hashlib.md5("".join(task).replace("#", "-").encode()).hexdigest()}{suffix}.yml'
                LOGGER.warning(f'Shortening file name from {file_name} to {short_file_name}.')
                file_name = short_file_name
            with (run / f'{finetune_dir}/configs/{file_name}').open('w') as f:
                f.write(task_finetune_cfg)


def create_gridsearch_dirs(run: Path,
                           combinations,
                           gridsearch_dir: Path,
                           inner_seeds: List[int],
                           n_trajectories: int,
                           query_size: int,
                           ft_epochs: Dict[float, int],
                           eval_every: List[int],
                           patience: int = None,
                           val_datasets: list = None,
                           tasks_ds_cfg: dict = None,
                           init_epoch: int = None,
                           n_dataset_evals: int = None,
                           optimizer: str = 'sgd',
                           classification_n_classes: dict = None,
                           pca_file_path: Path = None,
                           save_predictions: bool = True,
                           is_label_shared: bool = None,
                           batch_norm_mode: str = 'conventional'):
    abs_run = run.resolve()

    if init_epoch is None:
        best_epoch_file = run / 'best_epoch.txt'
        if not best_epoch_file.exists():
            print(f'best-epoch file {best_epoch_file} not found.')
            return
        with best_epoch_file.open('r') as fp:
            epoch = int(fp.read())
    else:
        epoch = init_epoch
    best_model_name = f'model_epoch{str(epoch).zfill(3)}.p'

    new_run_dirs = []
    for combination in tqdm(combinations):
        support = combination['support']
        lr = combination['lr']
        typ = combination['type']
        noise = combination['noise']
        spars = combination['sparsity']
        weight_decay = combination.get('weight_decay')
        pca_interpolate = combination.get('pca_interpolate')

        for i_seed in inner_seeds:
            combination_name = get_combination_name(combination, i_seed)
            run_dir = (abs_run / gridsearch_dir / combination_name).resolve()
            if run_dir.exists():
                continue
            run_dir.mkdir(parents=True)
            new_run_dirs.append(run_dir)
            os.symlink(abs_run / 'best_epoch.txt', run_dir / 'best_epoch.txt')
            os.symlink(abs_run / 'scaler.p', run_dir / 'scaler.p')
            os.symlink(abs_run / best_model_name, run_dir / best_model_name)

            cfg = Config(run / 'config.yml')
            if 'pca' in typ:
                if pca_file_path is None or not pca_file_path.exists():
                    raise ValueError(f'PCA not existing at {pca_file_path}.')
                cfg.update({'finetune_setup': 'pca',
                            'ig_pca_file': pca_file_path,
                            'pca_normalize': False})
                if spars is not None:
                    cfg.update({'pca_sparsity': spars})
                if 'noweight' in typ:
                    cfg.update({'use_pca_weights': False})
            elif 'normal' in typ or 'head' in typ or 'input' in typ:
                cfg.update({'finetune_setup': ''})
                if typ == 'head':
                    cfg.update({'finetune_modules': ['head']})
                elif typ == 'input':
                    cfg.update({'finetune_modules': {'lstm': ['bias_ih_l0', 'weight_ih_l0']}})
            else:
                raise ValueError('Unknown type')

            if val_datasets is None:
                val_datasets = cfg.train_datasets[::2]
            if n_dataset_evals is not None:
                cfg.update({'val_n_random_datasets': n_dataset_evals})
            if classification_n_classes is not None:
                cfg.update({'classification_n_classes': classification_n_classes})
            if is_label_shared is not None:
                cfg.update({'is_label_shared': is_label_shared})
            if pca_interpolate is not None:
                cfg.update({'pca_interpolation_factor': pca_interpolate,
                            'pca_normalize': True})
            optim = optimizer
            if optimizer in ['adam-pcaspace', 'adam-pcaspace-squared', 'adam-squared'] and 'pca' not in typ:
                optim = 'adam'
            if optimizer in ['sgd-squared'] and 'pca' not in typ:
                optim = 'sgd'
            cfg.update({'run_dir': run_dir.resolve(),
                        'finetune': True,
                        'finetune_epochs': ft_epochs[lr],
                        'finetune_lr': lr,
                        'finetune_early_stopping_patience': patience,
                        'plot_n_figures': 1,
                        'num_workers': 2,
                        'layer_per_dataset_eval': False,
                        'dataset_max_trajectories': n_trajectories,
                        'val_datasets': val_datasets,
                        'optimizer': optim,
                        'weight_decay': weight_decay if weight_decay is not None else 0.0,
                        'support_size': support,
                        'query_size': query_size,
                        'finetune_eval_every': eval_every,
                        'seed': i_seed,
                        'save_predictions': save_predictions,
                        'target_noise_std': noise,
                        'batch_norm_mode': batch_norm_mode})
            if 'adam' in typ and 'sgd' in optim:
                opt = 'adam'
                if 'pca' in typ and optim == 'sgd-squared':
                    opt = 'adam-squared'
                cfg.update({'optimizer': opt})
            if tasks_ds_cfg is not None:
                cfg.update(tasks_ds_cfg)
            cfg.dump_config(run_dir / 'config.yml')

    return new_run_dirs


def create_final_finetune_dirs(runs: List[Path],
                               support_sizes: List[int],
                               combinations,
                               inner_seeds: List[int],
                               gridsearch_dir: str,
                               test_tasks: List[str],
                               tasks_ds_cfg: dict = None,
                               gridsearch_seeds: List[int] = None,
                               best_ft_options=None,
                               best_ft_epochs=None,
                               n_results: int = 30,
                               split: str = 'test',
                               metric_name: str = 'mse',
                               metric_aggregation: str = 'median',
                               n_dataset_evals: int = None,
                               suffix: str = None):

    if best_ft_options is None:
        best_ft_options = {}
    if best_ft_epochs is None:
        best_ft_epochs = {}

    # inner_seeds: seeds to use for final finetuning.
    # gridsearch_seeds: seeds that were used in grid search
    if gridsearch_seeds is None:
        gridsearch_seeds = inner_seeds

    new_run_dirs = []
    global_results = {}
    for run in runs:
        for support in support_sizes:
            print(support)

            types = []
            for combination in combinations:
                if combination['support'] == support:
                    types.append((combination['type'], combination['noise'], combination['sparsity']))
            types = list(set(types))

            for (typ, noise, spars) in types:
                typ_name = f'{typ} noise {noise} {f" sparsity: {spars}" if spars is not None else ""}'
                # find best options/epochs for finetuning
                if (support, typ_name) not in best_ft_options:

                    type_best_option, type_best_epochs, type_global_results = \
                        get_best_options_epochs(runs,
                                                support,
                                                typ,
                                                noise,
                                                spars,
                                                n_results=n_results,
                                                combinations=combinations,
                                                inner_seeds=gridsearch_seeds,
                                                gridsearch_dir=gridsearch_dir,
                                                metric_aggregation=metric_aggregation,
                                                metric_name=metric_name)
                    best_ft_options[(support, typ_name)], best_ft_epochs[(support, typ_name)] = \
                        type_best_option, type_best_epochs
                    global_results[(support, typ_name)] = type_global_results
                else:
                    type_best_option, type_best_epochs = best_ft_options[(
                        support, typ_name)], best_ft_epochs[(support, typ_name)]

                # use the found best lr/epochs
                for i_seed in inner_seeds:
                    if type_best_option is None or type_best_epochs is None:
                        continue

                    gridsearch_best_combination = get_combination_name(best_ft_options[(support, typ_name)],
                                                                       gridsearch_seeds[0])
                    gridsearch_comb = (run.resolve() / gridsearch_dir / gridsearch_best_combination).resolve()

                    cfg = Config(run / gridsearch_dir / gridsearch_best_combination / 'config.yml')
                    cfg.update({'finetune_epochs': int(best_ft_epochs[(support, typ_name)]), 'seed': i_seed})

                    best_combination = get_combination_name(best_ft_options[(support, typ_name)], i_seed)
                    run_dir = (run.resolve() / _final_finetune_dir(gridsearch_dir, split, suffix) / best_combination).resolve()
                    if run_dir.exists():
                        continue
                    run_dir.mkdir(parents=True)
                    new_run_dirs.append(run_dir)
                    os.symlink(gridsearch_comb / 'best_epoch.txt', run_dir / 'best_epoch.txt')
                    for model in gridsearch_comb.glob('model_epoch*.p'):
                        os.symlink(model.resolve(), run_dir / model.name)
                    os.symlink(gridsearch_comb / 'scaler.p', run_dir / 'scaler.p')

                    if n_dataset_evals is not None:
                        cfg.update({'val_n_random_datasets': n_dataset_evals})

                    cfg.update({'finetune_eval_every': None,
                                'save_predictions': False,
                                'seed': i_seed,
                                'plot_n_figures': None,
                                'run_dir': run_dir,
                                'val_n_random_datasets': None,
                                f'{split}_datasets': test_tasks})
                    if tasks_ds_cfg is not None:
                        cfg.update(tasks_ds_cfg)
                    cfg.dump_config(run_dir / 'config.yml')

    for k, v in sorted(best_ft_options.items()):
        if v is None:
            continue
        print('support size:', str(k[0]).ljust(4), k[1].ljust(80),
              'LR:', str(v['lr']).ljust(6), 'epochs:', str(best_ft_epochs[k]).ljust(4),
              f'interpolate: {v["pca_interpolate"]}' if v.get("pca_interpolate") is not None else '')

    return new_run_dirs, best_ft_options, best_ft_epochs, global_results


def get_best_options_epochs(runs: List[Path],
                            support: int,
                            typ: str,
                            noise: float,
                            sparsity,
                            n_results: int,
                            combinations,
                            inner_seeds: List[int],
                            gridsearch_dir: str,
                            metric_aggregation: str,
                            metric_name: str = 'mse'):

    lower_is_better = metrics.lower_is_better(metric_name)
    best_option, best_epochs, best_option_metrics = None, None, np.inf
    if not lower_is_better:
        best_option_metrics = -np.inf

    global_results = pd.DataFrame()
    for combination in combinations:
        if support != combination['support'] or combination['type'] != typ \
                or combination['noise'] != noise or combination['sparsity'] != sparsity:
            continue

        option_metrics = {}
        for run in runs:
            for i_seed in inner_seeds:
                combination_name = get_combination_name(combination, i_seed)
                combination_metrics = _get_run_results(combination_name, run,
                                                       gridsearch_dir, i_seed, n_results, metric_aggregation)
                if len(combination_metrics) == 0:
                    return None, None, None
                option_metrics.update(combination_metrics)

        option_metrics = pd.DataFrame(option_metrics)
        option_metrics.columns.names = ['ds', 'run', 'seed']

        for c in option_metrics.columns:
            option_results = option_metrics[[c]]
            option_results.columns = pd.MultiIndex.from_tuples([tuple([combination_name.replace('_', ' ')] + list(c))])
            global_results = pd.concat([global_results, option_results], axis=1)

        if metric_aggregation in ['median', 'global']:
            option_metrics = option_metrics.median(axis=1, skipna=False)
        elif metric_aggregation == 'mean':
            option_metrics = option_metrics.mean(axis=1, skipna=False)
        else:
            raise ValueError("Unknown metric_aggregation")

        if (lower_is_better and option_metrics[option_metrics.idxmin()] < best_option_metrics) \
                or (not lower_is_better and option_metrics[option_metrics.idxmax()] > best_option_metrics):
            best_epochs = option_metrics.idxmin() if lower_is_better else option_metrics.idxmax()
            best_option = combination
            best_option_metrics = option_metrics[best_epochs]

    return best_option, best_epochs + 1, global_results


def _get_run_results(combination_name: str,
                     run: Path,
                     gridsearch_dir: str,
                     inner_seed: int,
                     n_results: int,
                     metric_aggregation: str) -> dict:
    run_metrics = {}

    results_dir = run / gridsearch_dir / combination_name
    if metric_aggregation in ['median', 'mean']:
        results_dir = results_dir / 'results' / 'finetune'
        if not results_dir.exists():
            print(f'No results for {combination_name, inner_seed}')
            return {}
        n_files = len(list(results_dir.glob('*.csv')))
        if n_files != n_results:
            print(f'Warning: {combination_name, inner_seed} has less results than expected ({n_files})')
            return {}
        for f in results_dir.glob('*.csv'):
            ds = f.name.replace('.csv', '')
            run_metrics[(ds, str(run.name), str(inner_seed))] = pd.read_csv(f, index_col=0)['0']
    elif metric_aggregation == 'global':
        results_dir = results_dir / 'predictions' / 'finetune'
        if not results_dir.exists():
            print(f'No results for {combination_name, inner_seed}')
            return {}
        n_files = len(list(results_dir.glob('*.nc')))
        if n_files != n_results:
            print(f'Warning: {combination_name, inner_seed} has less results than expected ({n_files})')
            return {}

        cfg = Config(run / 'config.yml')
        target_vars = [f'{var}_{target}' for var, offsets in cfg.target_vars['finetune'].items()
                       for target in offsets]
        metric_fn = metrics.get_metrics(cfg, target_vars, stds=None)[0]

        def _open_xr(f: Path):
            epoch = int(f.name.split('epoch')[-1][:-3])
            f_results = xr.open_dataset(f)
            return epoch, f_results['prediction'], f_results['target']
        all_predictions_targets = Parallel(n_jobs=32)(delayed(_open_xr)(f)
                                                      for f in list(results_dir.glob('*.nc')))
        all_predictions = {}
        all_targets = {}
        for epoch, predictions, targets in all_predictions_targets:  # type: ignore
            if epoch not in all_predictions:
                all_predictions[epoch] = []
                all_targets[epoch] = []
            all_predictions[epoch].append(predictions)
            all_targets[epoch].append(targets)

        def _get_metrics(epoch_pred, epoch_target):
            epoch_pred = torch.from_numpy(np.concatenate(epoch_pred))
            epoch_target = torch.from_numpy(np.concatenate(epoch_target))
            if cfg.classification_n_classes['finetune'] is not None:
                epoch_target = epoch_target.to(torch.long)  # xarray stores as int32
            return metric_fn(epoch_pred, epoch_target, 'global')
        epoch_results = Parallel(n_jobs=1, verbose=1)(delayed(_get_metrics)(epoch_pred,
                                                                            all_targets[epoch])  # type: ignore
                                                      for epoch, epoch_pred in all_predictions.items())
        epoch_results = dict(zip(all_predictions.keys(), epoch_results))  # type: ignore
        run_metrics[('global', str(run.name), str(inner_seed))] = pd.DataFrame(epoch_results).loc[cfg.metric[0]]
        run_metrics[('global', str(run.name), str(inner_seed))].sort_index(inplace=True)

    else:
        raise ValueError(f'Unknown aggregation strategy {metric_aggregation}')
    return run_metrics


def get_final_metrics(runs: List[Path],
                      noises: List[float],
                      support_sizes: List[int],
                      combinations,
                      best_ft_options,
                      inner_seeds: List[int],
                      query_size: int,
                      n_trajectories: int,
                      test_tasks: List[str],
                      gridsearch_dir: str,
                      metrics=None,
                      device: str = 'cpu',
                      split: str = 'test',
                      init_epoch: int = None,
                      metric_name: str = 'rmse',
                      metric_file_name: str = 'rmse',
                      metric_aggregation: str = 'median',
                      n_dataset_evals: int = None,
                      cfg_override=None,
                      no_ft_eval: bool = True, 
                      suffix: str = None):
    if metrics is None:
        metrics = {n: {} for n in noises}

    for run in runs:
        if init_epoch is None:
            best_epoch_file = run / 'best_epoch.txt'
            if not best_epoch_file.exists():
                print(f'best-epoch file {best_epoch_file} not found.')
                continue
            with best_epoch_file.open('r') as fp:
                best_epoch = str(int(fp.read())).zfill(3)
        else:
            best_epoch = str(init_epoch).zfill(3)

        if no_ft_eval:
            for noise in noises:
                for i_seed in inner_seeds:
                    if (support_sizes[0], 'no-finetune', (run.name, i_seed)) not in metrics[noise]:
                        cfg = Config(run / 'config.yml')
                        cfg.device = torch.device(device)
                        cfg.update({'finetune': False, 'seed': i_seed, 'target_noise_std': noise,
                                    'layer_per_dataset_eval': False, 'dataset_max_trajectories': n_trajectories,
                                    f'{split}_datasets': test_tasks, 'query_size': query_size,
                                    'num_workers': 2})
                        if cfg_override is not None:
                            cfg.update(cfg_override)
                        cfg.update({'val_n_random_datasets': n_dataset_evals})
                        no_ft_tester = get_tester(cfg, split=split)
                        no_ft_metrics, no_ft_global_metrics = no_ft_tester.evaluate(epoch=-1 if init_epoch is None
                                                                                    else init_epoch)
                        if metric_aggregation == 'global':
                            no_ft_metrics = no_ft_global_metrics.T
                        elif metric_aggregation in ['median', 'mean']:
                            pass
                        else:
                            raise ValueError(f'Unknown aggregation {metric_aggregation}')
                        for support in support_sizes:
                            metrics[noise][(support, 'no-finetune', (run.name, i_seed))] = no_ft_metrics
            torch.cuda.empty_cache()  # type: ignore

        for support in support_sizes:
            print(support)
            for noise in noises:
                types = []
                for combination in combinations:
                    if combination['support'] == support and combination['noise'] == noise:
                        types.append((combination['type'], combination['sparsity']))
                types = list(set(types))

                for (typ, spars) in types:
                    typ_name = f'{typ} noise {noise} {f" sparsity: {spars}" if spars is not None else ""}'
                    if all((support, typ_name, (run.name, i_seed)) in metrics for i_seed in inner_seeds):
                        continue

                    for i_seed in inner_seeds:
                        if (support, typ_name) not in best_ft_options or best_ft_options[(support, typ_name)] is None:
                            print('  No best epoch/options for', support, typ_name, i_seed)
                            continue
                        best_combination = get_combination_name(best_ft_options[(support, typ_name)], i_seed)
                        results_f = run / _final_finetune_dir(gridsearch_dir, split, suffix) / best_combination / \
                            'results'
                        if metric_aggregation == 'global':
                            results_f = results_f / f'global-{split}-model_epoch{best_epoch}-{metric_file_name}.csv'
                        elif metric_aggregation in ['median', 'mean']:
                            results_f = results_f / f'{split}-model_epoch{best_epoch}-{metric_file_name}.csv'
                        else:
                            raise ValueError(f'Unknown aggregation {metric_aggregation}')
                        if not results_f.is_file():
                            print('  No results for', results_f)
                            metrics[noise][(support, typ_name, (run.name, i_seed))] = pd.DataFrame(
                                {metric_name: {ds: np.nan for ds in test_tasks}})
                            continue
                        metrics[noise][(support, typ_name, (run.name, i_seed))] = pd.read_csv(results_f, index_col=0)
                        if metric_aggregation == 'global':
                            metrics[noise][(support, typ_name, (run.name, i_seed))] = \
                                metrics[noise][(support, typ_name, (run.name, i_seed))].T

    return metrics


def _final_finetune_dir(gridsearch_dir: str, split: str, suffix: str = None):
    final_finetune_dir = f'{gridsearch_dir}_finalFinetune'
    if split != 'test':
        final_finetune_dir += f'_{split}'
    if suffix is not None:
        final_finetune_dir += f'_{suffix}'
    return final_finetune_dir
