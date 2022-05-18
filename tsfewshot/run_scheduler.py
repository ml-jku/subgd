#!/usr/bin/env python
import argparse
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

LOGGER = logging.getLogger(__name__)


def _get_args() -> dict:

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval', 'finetune'])
    parser.add_argument('--directory', type=str, required=True)
    parser.add_argument('--epoch', type=str, required=False)
    parser.add_argument('--config-file', type=str, required=False)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=False)
    parser.add_argument('--gpu-ids', type=int, nargs='+', required=True)
    parser.add_argument('--runs-per-gpu', type=int, required=True)
    parser.add_argument('--name-filter', type=str, nargs='+', required=False)
    parser.add_argument('--exclude', type=str, nargs='+', required=False)

    args = vars(parser.parse_args())

    args['directory'] = Path(args['directory'])
    if not args['directory'].is_dir():
        raise ValueError(f'No folder at {args["directory"]}')

    return args


def _main():
    args = _get_args()
    schedule_runs(**args)


def schedule_runs(mode: str, directory: Path, gpu_ids: List[int], runs_per_gpu: int,
                  epoch: int = None, split: str = None, config_file: str = None,
                  name_filter: list = None, exclude: list = None):
    """Schedule multiple runs across one or multiple GPUs.

    Parameters
    ----------
    mode : {'train', 'eval', 'finetune'}
        Use 'train' if you want to schedule training of multiple models, 'evaluate' if you want to schedule
        evaluation of multiple trained models and 'finetune' if you want to schedule finetuning with multiple configs.
    directory : Path
        If mode is 'train', this path should point to a folder containing the config files (.yml)
        to use for model training. For each config file, one run is started. If mode is 'eval', this path
        should point to the folder containing the different model run directories.
    gpu_ids : List[int]
        List of GPU ids to use for training/evaluating.
    runs_per_gpu : int
        Number of runs to start on a single GPU.
    epoch : int, optional
        Epoch to evaluate. -1 to use the best epoch as stored in run_dir/best_epoch.txt
    split : {'train', 'val', 'test'}, optional
        Data split to run evaluation on.
    config_file : str, optional
        Config file to pass to all evaluation runs to update the original config (all runs will receive the same file).
    name_filter : list, optional
        List of strings by which to filter names of run configurations/directories.
    exclude : list, optional
        List of strings to exclude from of run configurations/directories.
    """

    if mode in ['train', 'finetune']:
        if name_filter is None:
            processes = list(directory.glob('**/*.yml'))
        else:
            processes = []
            for f in name_filter:
                processes += list(directory.glob(f'**/*{f}*.yml'))
        if exclude is not None:
            for exclude_name in exclude:
                processes = [p for p in processes if exclude_name not in p.stem]
        processes = list(set(processes))
        if split is not None:
            raise ValueError('Cannot specify --split during training/finetuning.')
        if config_file is not None:
            raise ValueError('Cannot specify --config-file during training/finetuning. Use --directories.')
    elif mode == 'eval':
        if name_filter is None:
            processes = list(directory.glob('*'))
        else:
            processes = []
            for f in name_filter:
                processes += list(directory.glob(f'*{f}*'))
        if exclude is not None:
            for exclude_name in exclude:
                processes = [p for p in processes if exclude_name not in p.name]
        processes = list(set(processes))
    else:
        raise ValueError('"mode" must be either "train", "finetune", or "eval"')

    # if used as command line tool, we need full path's to the fils/directories
    processes = [str(p.absolute()) for p in processes]

    # for approximately equal memory usage during hyperparam tuning, randomly shuffle list of processes
    random.shuffle(processes)

    # array to keep track on how many runs are currently running per GPU
    n_parallel_runs = len(gpu_ids) * runs_per_gpu
    gpu_counter = np.zeros((len(gpu_ids)), dtype=int)

    # for command line tool, we need full path to the main.py script
    script_path = str(Path(__file__).absolute().parent / 'run.py')

    running_processes = {}
    counter = 0
    while True:

        # start new runs
        for _ in range(n_parallel_runs - len(running_processes)):

            if counter >= len(processes):
                break

            # determine which GPU to use
            node_id = np.argmin(gpu_counter)
            gpu_counter[node_id] += 1
            gpu_id = gpu_ids[node_id]
            process = processes[counter]

            # start run via subprocess call
            if mode in ['train', 'finetune']:
                run_command = f'python {script_path} {mode} --config-file {process} --gpu {gpu_id}'
            elif mode == 'eval':
                if Path(process).is_dir():
                    run_command = f'python {script_path} {mode} --run-dir {process} --gpu {gpu_id}'
                elif Path(process).is_file():
                    run_command = f'python {script_path} {mode} --config-file {process} --gpu {gpu_id}'
                else:
                    raise ValueError(f'{process} is neither file nor folder: skipping.')
                if epoch is not None:
                    run_command += f' --epoch {epoch}'
                if split is not None:
                    run_command += f' --split {split}'
                if config_file is not None:
                    run_command += f' --config-file {config_file}'
            else:
                raise ValueError(f'Unknown mode {mode}')
            LOGGER.info(f'Starting run {counter+1}/{len(processes)}: {run_command}')
            running_processes[(counter + 1, run_command, node_id)] = subprocess.Popen(run_command,
                                                                                      stdout=subprocess.DEVNULL,
                                                                                      shell=True)

            counter += 1
            time.sleep(2)

        # check for completed runs
        for key, process in running_processes.items():
            if process.poll() is not None:
                LOGGER.info(f'Finished run {key[0]} ({key[1]})')
                gpu_counter[key[2]] -= 1
                LOGGER.info('Cleaning up...\n')
                try:
                    _ = process.communicate(timeout=5)
                except TimeoutError:
                    LOGGER.warning('')
                    LOGGER.warning(f'WARNING: PROCESS {key} COULD NOT BE REAPED!')
                    LOGGER.warning('')
                running_processes[key] = None

        # delete possibly finished runs
        running_processes = {key: val for key, val in running_processes.items() if val is not None}
        time.sleep(2)

        if (len(running_processes) == 0) and (counter >= len(processes)):
            break

    LOGGER.info('Done')
    sys.stdout.flush()


if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                        level=logging.INFO, format='%(asctime)s: %(message)s')

    # Log uncaught exceptions
    def exception_logging(typ, value, traceback):
        """Make sure all exceptions are logged. """
        LOGGER.exception('Uncaught exception', exc_info=(typ, value, traceback))

    sys.excepthook = exception_logging

    LOGGER.info(f'Command: python {" ".join(sys.argv)}\n')
    _main()
