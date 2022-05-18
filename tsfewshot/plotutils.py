import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def visualize_gridsearch_results_dataframe(gridsearch_results):
    def get_lr_alpha_from_column(exp_name: str, lr_prefix='lr', alpha_prefix='pcainterpolate'):
        """returns lr and alpha from the experiment name given as string"""
        lr = 0
        alpha = 0
        #exp_name = column[0]
        exp_name_tokens = exp_name.split(' ')
        for t in exp_name_tokens:
            if t.find(lr_prefix) == 0:
                lr = float(t[len(lr_prefix):])
            elif t.find(alpha_prefix) == 0:
                alpha = float(t[len(alpha_prefix):])

        return lr, alpha

    n_plots = len([0 for x in gridsearch_results.values() if x is not None]) * 2
    f, ax = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True, sharey=True)
    if n_plots == 1:
        ax = [ax]
    i = 0
    for type_name, type_results in gridsearch_results.items():
        if type_results is None:
            continue
        type_results = type_results.copy()  # dataframe with epoch accuracies as rows and different gridsearch runs as columns
        type_results.columns = pd.MultiIndex.from_tuples(type_results.columns, names=['type', 'dataset', 'run', 'seed'])
        type_results = type_results.groupby('type', axis=1).median()

        max_accuracies = [(type_results[c].max(), type_results[c].argmax())
                          for c in type_results.columns]  # contains tuples (accuracy, epoch) for every run
        max_accuracies_df = pd.DataFrame([max_accuracies], columns=type_results.columns)
        lr_alpha_acc_dict = {get_lr_alpha_from_column(
            c): max_accuracies_df.loc[0, c] for c in list(max_accuracies_df.columns)}

        # plot accuracy heatmap
        lr_alpha_df = pd.DataFrame()
        for k, v in lr_alpha_acc_dict.items():
            lr_alpha_df.loc[k[0], k[1]] = v[0]  # accuracy
        lr_alpha_df = lr_alpha_df.sort_index(axis=0).sort_index(axis=1)

        sns.heatmap(lr_alpha_df.to_numpy(), xticklabels=lr_alpha_df.columns,
                    yticklabels=lr_alpha_df.index, annot=True, fmt='.4g',
                    cbar_kws={'label': 'max. accuracy over epochs'}, ax=ax[i])
        ax[i].legend(frameon=False, loc=(1, 0))
        ax[i].set_title(type_name)
        ax[i].set_xlabel('alphas')
        ax[i].set_ylabel('learning rates')
        i += 1

        # plot epoch heatmap
        lr_alpha_df = pd.DataFrame()
        for k, v in lr_alpha_acc_dict.items():
            lr_alpha_df.loc[k[0], k[1]] = int(v[1])  # epoch
        lr_alpha_df = lr_alpha_df.sort_index(axis=0).sort_index(axis=1)

        sns.heatmap(lr_alpha_df.to_numpy(), xticklabels=lr_alpha_df.columns,
                    yticklabels=lr_alpha_df.index, annot=True, fmt='.0f',
                    cbar_kws={'label': 'epoch of max. accuracy'}, ax=ax[i])
        ax[i].legend(frameon=False, loc=(1, 0))
        ax[i].set_title(type_name)
        ax[i].set_xlabel('alphas')
        ax[i].set_ylabel('learning rates')
        i += 1

    return f


def visualize_gridsearch_finetune_run(run: Union[str, Path], gridsearch_dir: str, gridsearch_run_dir: Union[str, List[str]]):
    # load all csv files with pandas
    # in /results/finetune
    run = Path(run)
    if not isinstance(gridsearch_run_dir, list):
        gridsearch_run_dir = [gridsearch_run_dir]

    n_plots = len(gridsearch_run_dir)
    fig, ax = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True, sharey=True)
    if n_plots == 1:
        ax = [ax]

    i = 0
    for gd in gridsearch_run_dir:
        r = run / gridsearch_dir / gd

        results_finetune_dir = r / "results" / "finetune"

        finetune_results = []
        assert results_finetune_dir.exists()
        for f in results_finetune_dir.iterdir():
            res = pd.read_csv(f)
            res.columns = pd.MultiIndex.from_arrays(
                [[f.stem, f.stem], ["epoch", "accuracy"]], names=('classes', 'results'))
            finetune_results.append(res)

        fr = pd.concat(finetune_results, axis=1)
        for task in fr.columns.unique(level=0):
            tdf = fr[task]  # task dataframe
            ax[i].plot(tdf["epoch"], tdf["accuracy"], label=task)

        ax[i].legend(frameon=False, loc=(1, 0))
        ax[i].set_title(gd)
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel("accuracy")
        i += 1

    return fig

def compare_gridsearch_finetune_runs(run: Union[str, Path], gridsearch_dir: str, gridsearch_run_dir: Union[str, List[str]], n_classes_per_plot: int = 5):
    # load all csv files with pandas
    # in /results/finetune
    run = Path(run)
    if not isinstance(gridsearch_run_dir, list):
        gridsearch_run_dir = [gridsearch_run_dir]

    n_plots = n_classes_per_plot
    fig, ax = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True, sharey=True)
    if n_plots == 1:
        ax = [ax]

    # load data from dirs
    grd_frs_dict = {} # gridsearch run dir - finetune runs - dict
    for grd in gridsearch_run_dir:
        r = run / gridsearch_dir / grd

        results_finetune_dir = r / "results" / "finetune"

        finetune_results = []
        assert results_finetune_dir.exists()
        for f in results_finetune_dir.iterdir():
            res = pd.read_csv(f)
            res.columns = pd.MultiIndex.from_arrays(
                [[f.stem, f.stem], ["epoch", "accuracy"]], names=('classes', 'results'))
            finetune_results.append(res)

        fr = pd.concat(finetune_results, axis=1)
        
        grd_frs_dict[grd] = fr

    # print(grd_frs_dict)
    # get n_classes_per_plot tasks from a run <- these will be used for plotting

    fr = grd_frs_dict[gridsearch_run_dir[0]]
    tasks_to_plot = fr.columns.unique(level=0)[:n_classes_per_plot]
    
    i = 0
    for task in tasks_to_plot:
        for grd, fr in grd_frs_dict.items():          
            tdf = fr[task]  # task dataframe
            ax[i].plot(tdf["epoch"], tdf["accuracy"], label=grd)
        ax[i].legend(frameon=False, loc=(1, 0))
        ax[i].set_title(task)
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel("accuracy")
        i += 1

    return fig