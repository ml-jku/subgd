import copy
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset, Subset

from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset
from tsfewshot.data.hbvedudataset import HBVEduDataset
from tsfewshot.data.rlcdataset import RLCDataset
from tsfewshot.data.sinusdataset import SinusDataset
from tsfewshot.data.utils import IndexSampler
from tsfewshot.utils import get_random_states, set_seed


class EpisodicDataset(IterableDataset):
    """Dataset to combine multiple datasets into meta-learning episodes.

    For N-way K-shot settings, each dataset corresponds to a class. In each episode, N datasets are selected, and from
    each dataset K samples are loaded.
    For non-label-sharing settings, the labels of the individual datasets are discarded and replaced by integers
    between 0 and N.

    Parameters
    ----------
    cfg : Config
        The run configuration
    datasets: Dict[str, data.Dataset]
        The individual datasets that will be combined into episodes.
    query_size : int
        Size of the generated query sets.
    """

    def __init__(self, cfg: Config, datasets: Dict[str, BaseDataset], query_size: int):
        super().__init__()
        self._cfg = cfg
        self._datasets = datasets
        self._dataset_names = list(self._datasets.keys())
        self._n_way = cfg.classification_n_classes['train']
        if self._n_way is None:
            self._n_way = 1  # regression is like 1-way classification

        self._query_loaders = {}
        self._support_loaders = {}
        for ds_name, dataset in datasets.items():
            self._query_loaders[ds_name] = DataLoader(dataset,
                                                      batch_size=query_size,
                                                      shuffle=True,
                                                      num_workers=0)
            self._support_loaders[ds_name] = DataLoader(dataset,
                                                        batch_size=cfg.support_size,
                                                        shuffle=True,
                                                        num_workers=0)

    def __len__(self) -> int:
        return sys.maxsize

    def __iter__(self):
        while True:
            support_set = []
            query_set = []

            random_task = next(self._datasets[list(self._datasets.keys())[0]].get_valid_dataset_combinations(
                list(self._datasets.keys()), self._n_way, n_random_combinations=1))

            if self._cfg.meta_dataset == 'sinusoid':
                # draw new parameters each time a task is used
                for ds in self._datasets.values():
                    ds.draw_parameters()  # type: ignore

            for i, class_name in enumerate(random_task):
                # creating a new iter is not a performance issue because we use persistent workers.
                task_support = next(iter(self._support_loaders[class_name]))
                task_query = next(iter(self._query_loaders[class_name]))

                # in non-shared label settings, we need to create new labels for each class
                if self._n_way > 1:
                    task_support['y'] = torch.full_like(task_support['y'], i, dtype=torch.long)
                    task_query['y'] = torch.full_like(task_query['y'], i, dtype=torch.long)
                support_set.append(task_support)
                query_set.append(task_query)

            support_set = {k: torch.cat([support_set[i][k] for i in range(len(support_set))], dim=0)
                           for k in support_set[0].keys() if k != 'dataset'}
            query_set = {k: torch.cat([query_set[i][k] for i in range(len(query_set))], dim=0)
                         for k in query_set[0].keys() if k != 'dataset'}
            support_set['dataset'] = '-'.join(random_task)  # type: ignore
            query_set['dataset'] = '-'.join(random_task)  # type: ignore

            yield support_set, query_set


class EpisodicOdeDataset(IterableDataset):
    """Dataset to combine multiple ODE datasets into meta-learning episodes.

    Extracts (shorter) sequences of the dataset as support and query set. 
    Number of timesteps is support size or query size.

    Parameters
    ----------
    cfg : Config
        The run configuration
    datasets: Dict[str, data.Dataset]
        The individual datasets that will be combined into episodes.
    query_size : int
        Size of the generated query sets.
    """

    def __init__(self, cfg: Config, datasets: Dict[str, BaseDataset], query_size: int):
        super().__init__()
        self._cfg = cfg
        self._datasets = datasets
        self._dataset_names = list(self._datasets.keys())
        self._n_way = cfg.classification_n_classes['train']
        if self._n_way is None:
            self._n_way = 1  # regression is like 1-way classification

        if self._cfg.meta_dataset != 'rlc':
            raise ValueError('Episodic ODE dataset only supported for RLC dataset.')
        
        if self._cfg.query_size != query_size:
            raise ValueError('Config argument `maml_inner_batch_size` must be equal to `query_size`')

        if self._cfg.support_size != self._cfg.seq_length or self._cfg.query_size != self._cfg.seq_length:       
            raise ValueError(f'Query and support size must have length {self._cfg.seq_length}')



        self._support_query_loaders = {}
        for ds_name, dataset in datasets.items():
            # here batch_size is 1 as the support and query size are defined as length of the sequence
            self._support_query_loaders[ds_name] = DataLoader(dataset,
                                                      batch_size=1, 
                                                      shuffle=True,
                                                      num_workers=0)

    def __len__(self) -> int:
        return sys.maxsize

    def __iter__(self):
        while True:
            support_set = []
            query_set = []

            random_task = next(self._datasets[list(self._datasets.keys())[0]].get_valid_dataset_combinations(
                list(self._datasets.keys()), self._n_way, n_random_combinations=1))

            for i, class_name in enumerate(random_task):
                # creating a new iter is not a performance issue because we use persistent workers.
                support_set = next(iter(self._support_query_loaders[class_name]))
                
                # rejection sampling for query set
                # make sure that query and support are never overlapping
                while True:
                    query_set = next(iter(self._support_query_loaders[class_name]))

                    if support_set['sample'] != query_set['sample']:
                        break # support and query set cannot overlap in this case
                    else:
                        if query_set['offset'] > support_set['offset']:
                            offset_distance = query_set['offset'] - support_set['offset']
                        else:
                            offset_distance = support_set['offset'] - query_set['offset']
                        if offset_distance > self._cfg.seq_length:
                            # no overlap
                            break


            support_set['dataset'] = '-'.join(random_task)  # type: ignore
            query_set['dataset'] = '-'.join(random_task)  # type: ignore

            yield support_set, query_set


class EpisodicTestDataset(Dataset):
    """Dataset to combine multiple datasets into meta-testing episodes (one support set and many query sets).

    For N-way K-shot settings, each dataset corresponds to a class. In each episode, N datasets are selected, and from
    each dataset K samples are loaded.
    For non-label-sharing settings, the labels of the individual datasets are discarded and replaced by integers
    between 0 and N.

    Parameters
    ----------
    cfg : Config
        The run configuration
    datasets : Dict[str, data.Dataset]
        The individual datasets that will be combined into episodes.
    seed : int, optional
        Seed to use to draw support and query set. If None, will draw randomly without fixed seed.
    """

    def __init__(self, cfg: Config, datasets: Dict[str, BaseDataset], seed: int = None):
        super().__init__()
        self.name = '-'.join(datasets.keys())
        if seed is not None:
            self.name += f'/{seed}'
        self._cfg = cfg
        self._datasets = datasets
        self._is_label_shared = cfg.is_label_shared
        self._n_way = cfg.classification_n_classes['finetune']
        if self._n_way is None:
            self._n_way = 1  # regression is like 1-way classification
        self._dataset_to_label = {}

        self._seed = seed
        self._support_set = self._generate_support_and_query()

    def get_support_set(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a support set consisting of inputs, targets, sample ids, and offsets. """
        return self._support_set  # type: ignore

    def _generate_support_and_query(self):
        query_datasets = []
        self._dataset_to_label = {}
        support_set = []

        # these variables represent one batch, which is used for finetuning
        support_x, support_y, support_sample_ids, support_offsets, support_y_base = None, None, None, None, None

        for i, (ds_name, dataset) in enumerate(self._datasets.items()):
            self._dataset_to_label[ds_name] = i
            dataset = self._datasets[ds_name]

            if isinstance(dataset, RLCDataset):
                if self._cfg.train_val_split is not None:
                    raise ValueError("train_val_split is not supported for this dataset.")
                support_size = self._cfg.support_size
                raw_data_x = dataset._x[ds_name]
                raw_data_y = dataset._y[ds_name]
                raw_data_y_base = dataset._y_base[ds_name]

                if support_size > 0:
                    # make sure support size not greater than trajectory length
                    assert support_size <= raw_data_x.shape[1], "Support size cannot be longer than trajectory length."
                    #! So far only first trajectory supported
                    # i.e. take the first num_timesteps = support_size steps from the first trajectory
                    # (i.e. there must be at least one)
                    # TODO support different trajectories (sample_ids and offsets) as support samples
                    sample_ids = 0
                    offsets = 0
                    # get support sample here
                    support_x = {'x0': torch.zeros(size=(1, 1, len(self._cfg.eulerode_config['state_space']))),
                                 'u': raw_data_x[[sample_ids], offsets:support_size]}
                    support_y = raw_data_y[[sample_ids], offsets:support_size]
                    support_sample_ids = sample_ids
                    support_offsets = offsets
                    support_y_base = raw_data_y_base[[sample_ids], offsets:support_size]

                # iterate over trajectory dimensions
                # take trajectories except the trajactory where the support set is taken from as query set

                # this is the number of timesteps in the trajectory taken from the dataset
                # when the query size smaller than the trajectory length the query set contains only a single trajectory
                # when the query size is greater than the trajectory length the query set contains only full trajectories
                # (it is always a multiple of the trajectory length)
                # this means the actual query size can deviate from the specified query size in the config
                query_size = self._cfg.query_size
                sample_id = 1
                assert raw_data_x.shape[0] > 1, f"The dataset contains {raw_data_x.shape[0]} trajectories. Need at least two to take support and query set."
                query_offset_start = 0  # start from beginning of new trajectory
                state_space = self._cfg.eulerode_config['state_space']

                if query_size < raw_data_x.shape[1]:  # trajectory length
                    query_offset_stop = query_size
                    x0 = torch.zeros(size=(1, len(state_space)))

                    query_x = {'x0': x0, 'u': raw_data_x[sample_id][query_offset_start:query_offset_stop]}

                    query_y = raw_data_y[sample_id][query_offset_start:query_offset_stop]
                    query_y_base = raw_data_y_base[sample_id][query_offset_start:query_offset_stop]
                    query_dataset = ds_name
                    query_sample = sample_id

                    query_item = {'x': query_x, 'y': query_y, 'y_base': query_y_base,
                                  'dataset': query_dataset, 'sample': query_sample, 'offset': query_offset_start}
                    query_datasets.append(query_item)
                else:
                    query_offset_stop = raw_data_x.shape[1]
                    num_query_trajectories = min(int(query_size / raw_data_x.shape[1]), raw_data_x.shape[0] - 1)

                    for j in range(0, num_query_trajectories):
                        sample_id += j
                        x0 = torch.zeros(size=(1, len(state_space)))
                        query_x = {'x0': x0, 'u': raw_data_x[sample_id][query_offset_start:query_offset_stop]}
                        query_y = raw_data_y[sample_id][query_offset_start:query_offset_stop]
                        query_y_base = raw_data_y_base[sample_id][query_offset_start:query_offset_stop]
                        query_dataset = ds_name
                        query_sample = sample_id

                        query_item = {'x': query_x, 'y': query_y, 'y_base': query_y_base,
                                      'dataset': query_dataset, 'sample': query_sample, 'offset': query_offset_start}
                        query_datasets.append(query_item)

                self._query_dataset = query_datasets  # this is a List[Dict[str, Any]]
            else:
                old_random_state = None
                if self._seed is not None:
                    old_random_state = get_random_states()
                    set_seed(self._seed)

                if isinstance(dataset, HBVEduDataset):
                    all_indices = list(range(len(dataset)))
                    all_query_indices = all_indices[dataset.support_ranges[ds_name]:]
                    all_support_indices = all_indices[:dataset.support_ranges[ds_name]]
                    shuffled_query_indices = np.random.permutation(all_query_indices)
                    shuffled_support_indices = np.random.permutation(all_support_indices)
                    n_query_samples = min(self._cfg.query_size, len(dataset) - dataset.support_ranges[ds_name])
                    query_indices = shuffled_query_indices[:n_query_samples]
                    support_indices = shuffled_support_indices[:self._cfg.support_size]
                    support_dataset = dataset
                else:
                    if isinstance(dataset, SinusDataset):
                        # draw new parameters each time a dataset is used
                        # have to copy the dataset because we generate the support set now and the query set later.
                        # Without copying, by the time we'd generate the query set, the parameters would already
                        # have changed again.
                        dataset = copy.deepcopy(dataset)
                        dataset.draw_parameters()

                    shuffled_indices = np.random.permutation(len(dataset))
                    if old_random_state is not None:
                        # restore seed
                        set_seed(old_random_state)

                    # take query indices from the front, so they are independent of the support size
                    n_query_samples = min(self._cfg.query_size, len(dataset) - self._cfg.support_size)
                    query_indices = shuffled_indices[:n_query_samples]
                    support_indices = shuffled_indices[n_query_samples:n_query_samples + self._cfg.support_size]
                    support_dataset = dataset

                query_datasets.append(Subset(dataset, query_indices))

                if len(support_indices) > 0:
                    task_support = next(iter(DataLoader(support_dataset,
                                                        batch_size=len(support_indices),
                                                        sampler=IndexSampler(support_indices),
                                                        num_workers=0)))

                    # in non-shared label settings, we need to create new labels for each class
                    if not self._is_label_shared:
                        task_support['y'] = torch.full_like(task_support['y'],
                                                            self._dataset_to_label[ds_name],
                                                            dtype=torch.long)
                    support_set.append(task_support)

                self._query_dataset = ConcatDataset(query_datasets)

                if self._cfg.support_size > 0:
                    support_x = torch.cat([task_support['x'] for task_support in support_set], dim=0)
                    support_y = torch.cat([task_support['y'] for task_support in support_set], dim=0)
                    support_sample_ids = torch.cat([task_support['sample'] for task_support in support_set], dim=0)
                    support_offsets = torch.cat([task_support['offset'] for task_support in support_set], dim=0)
                    support_y_base = torch.cat([task_support['y_base'] for task_support in support_set], dim=0)

        return support_x, support_y, support_sample_ids, support_offsets, support_y_base

    def __len__(self) -> int:
        return len(self._query_dataset)

    def __getitem__(self, index: int):
        query_set = self._query_dataset[index]

        if not self._is_label_shared:
            query_set['y'] = torch.full_like(query_set['y'], self._dataset_to_label[query_set['dataset']])
        return query_set
