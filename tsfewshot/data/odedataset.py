# pylint: disable=invalid-name
import logging
from typing import Dict, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from numpy import cos, sin

from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class ODEDatasetSimple(BaseDataset):
    """Dataset for simple 1D ODE trajectories which were generated with ODEDatasetGenerator.

    For details on the generating process, see :func:`tsfewshot.data.odeutils.generate_simple_ode_dataset`.

    Parameters
    ----------
    cfg : Config
        Run configuration
    split : {'train', 'val', 'test'}
        Period for which the dataset will be used.
    dataset : str, optional
        If provided, the dataset will ignore the settings in `cfg` and use this dataset instead.
    is_train : bool, optional
        Indicates whether the dataset will be used for training or evaluation (including finetuning).
    train_scaler : Dict[str, Dict[str, torch.Tensor]], optional
        Pre-calculated scaler to use for normalization of input/output values.
    train_val_split : float, optional
        float between 0 and 1 to subset the created dataset. If provided, the created dataset will hold a dictionary
        mapping each dataset name to the indices used in the train split. Subsequently, these indices can be used
        to subset the corresponding validation datasets.
    silent : bool, optional
        Option to override cfg.silent.
    """

    def __init__(self, cfg: Config,
                 split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):
        self.data = None
        super().__init__(cfg, split, dataset=dataset, is_train=is_train, train_scaler=train_scaler,
                         train_val_split=train_val_split, silent=silent)

    def _load_dataset(self, cfg: Config, dataset: str) -> Optional[xr.Dataset]:
        train_file = cfg.base_dir / f'{dataset}'
        if not train_file.exists():
            LOGGER.warning(f'Dataset {dataset} has no npy file. skipping.')
            return None

        all_data = np.load(train_file)
        self.data = all_data['data']

        # take first 100 timesteps
        self.data = self.data[:, 0:100]

        if self.data.shape[1] != 100:
            LOGGER.warning(f'Dataset timeseries {dataset} have length {self.data.shape[1]} != 100. Skipping.')
            return None

        if np.isnan(self.data).any():
            LOGGER.warning(f'Dataset {dataset} has NaNs. Skipping.')
            return None

        xarray = xr.Dataset({'x': (['sample', 'step'], self.data)})

        return xarray

    def visualize(self, n_samples2vis: int = 10):
        """Create a plot of `n_samples2vis` randomly chosen trajectories of the loaded dataset.

        Parameters
        ----------
        n_samples2vis : int, optional
            Controls how many trajectories are randomly chosen from the dataset. Defaults to 10.
        """
        n_samples, _ = self.data.shape
        chosen_idcs = np.random.choice(np.arange(n_samples), size=n_samples2vis)

        plt.figure()
        for cidx in chosen_idcs:
            plt.plot(self.data[cidx, :])
        plt.show()


class DoublePendulum(BaseDataset):
    """Dataset for 4D trajectories of a double pendulum.

    Datasets can be generated with :class:`tsfewshot.data.odeutils.ODEDatasetGenerator`.
    For details on the generating process, see :func:`tsfewshot.data.odeutils.generate_double_pendulum_dataset`.

    * th1 - is the angle (in radians) of displacement of the first pendulum
    * w1  - is the angle velocity of the first mass
    * th2 - is the angle (in radians) of displacement of the second pendulum
    * w2  - is the angle velocity of the second mass

    Parameters
    ---------
    cfg : Config
        Run configuration
    split : {'train', 'val', 'test'}
        Period for which the dataset will be used.
    dataset : str, optional
        If provided, the dataset will ignore the settings in `cfg` and use this dataset instead.
    is_train : bool, optional
        Indicates whether the dataset will be used for training or evaluation (including finetuning).
    train_scaler : Dict[str, Dict[str, torch.Tensor]], optional
        Pre-calculated scaler to use for normalization of input/output values.
    train_val_split : float, optional
        float between 0 and 1 to subset the created dataset. If provided, the created dataset will hold a dictionary
        mapping each dataset name to the indices used in the train split. Subsequently, these indices can be used
        to subset the corresponding validation datasets.
    silent : bool, optional
        Option to override cfg.silent.
    """

    def __init__(self, cfg: Config,
                 split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):
        self.data = None
        self.params = None
        super().__init__(cfg, split, dataset=dataset, is_train=is_train, train_scaler=train_scaler,
                         train_val_split=train_val_split, silent=silent)

    def _load_dataset(self, cfg: Config, dataset: str) -> Optional[xr.Dataset]:
        train_file = cfg.base_dir / f'{dataset}'
        if not train_file.exists():
            LOGGER.warning(f'Dataset {dataset} has no npy file. skipping.')
            return None

        all_data = np.load(train_file, allow_pickle=True)

        self.data = all_data['data']
        self.params = all_data['params'].item()

        for name, (min_, max_) in cfg.filter_samples.items():
            idx = ['th1', 'w1', 'th2', 'w2'].index(name)
            self.data = self.data[np.all((self.data[:, :, idx] >= min_) & (self.data[:, :, idx] <= max_), axis=1)]
        if self.data.shape[0] == 0:
            LOGGER.warning(f'Dataset {dataset} has not samples. Skipping.')
            return None

        # take first 100 timesteps
        self.data = self.data[:, 0:100]

        if self.data.shape[1] != 100:
            LOGGER.warning(f'Dataset timeseries {dataset} have length {self.data.shape[1]} != 100. Skipping.')
            return None

        if np.isnan(self.data).any():
            LOGGER.warning(f'Dataset {dataset} has NaNs. Skipping.')
            return None

        xarray = xr.Dataset({'th1': (['sample', 'step'], self.data[:, :, 0]),
                             'w1': (['sample', 'step'], self.data[:, :, 1]),
                             'th2': (['sample', 'step'], self.data[:, :, 2]),
                             'w2': (['sample', 'step'], self.data[:, :, 3])
                             })
        for param, value in self.params.items():
            xarray[param] = (('sample', 'step'), np.full(self.data.shape[:2], value))

        return xarray

    def visualize(self, nfigures_perside: int = 3, show_fig: bool = True) -> animation.FuncAnimation:
        """Create an animation with ``nfigures_perside**2`` randomly chosen double pendulums of the loaded dataset.

        To display the animation in a Jupyter notebook, use ``show_fig = False``:

        .. code-block:: python

            anim = ds.visualize(show_fig=False)
            display(IPython.display.HTML(anim.to_html5_video()))  # show the animation
            plt.clf()  # prevent matplotlib from displaying the un-animated figure

        Parameters
        ----------
        nfigures_perside : int, optional
            Controls how many pendulums are animated. Defaults to 3.
        show_fig : bool, optional
            If False, will not call plt.show(). Useful when called from a Jupyter notebook.

        Returns
        -------
        animation.FuncAnimation
            Animated pendulum plots.
        """

        n_samples2vis = nfigures_perside**2
        n_samples, n_steps, _ = self.data.shape
        chosen_idcs = np.random.choice(np.arange(n_samples), size=n_samples2vis)

        L1, L2 = self.params['l1'], self.params['l2']
        chosen_traj = []
        xmin, xmax, ymin, ymax = 0, 0, 0, 0
        for cidx in chosen_idcs:
            traj = self.data[cidx, :, :]
            x1 = L1 * sin(traj[:, 0])
            y1 = -L1 * cos(traj[:, 0])

            x2 = L2 * sin(traj[:, 2]) + x1
            y2 = -L2 * cos(traj[:, 2]) + y1
            xmin = min([min(x1), min(x2), xmin])
            xmax = max([max(x1), max(x2), xmax])
            ymin = min([min(y1), min(y2), ymin])
            ymax = max([max(y1), max(y2), ymax])
            chosen_traj.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        fig, axes = plt.subplots(nrows=nfigures_perside, ncols=nfigures_perside, figsize=(10, 10))
        for ax in axes.flat:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.grid()

        lines, paths = [], []
        for num, (cidx, ax) in enumerate(zip(chosen_idcs, axes.flat)):
            line, = ax.plot([], [], 'o-', color=f'C{num:d}', lw=2)
            lines.append(line)
            path, = ax.plot([], [], color=f'C{num:d}', alpha=0.5)
            paths.append(path)

        step_template = 'step = {:d}'
        step_text = fig.suptitle('')

        def init():
            for line in lines:
                line.set_data([], [])

            for path in paths:
                path.set_data([], [])

            step_text.set_text('')
            return lines + paths

        def animate(ii, path_len=100):
            for traj, line, path in zip(chosen_traj, lines, paths):
                cur_x = [0, traj['x1'][ii], traj['x2'][ii]]
                cur_y = [0, traj['y1'][ii], traj['y2'][ii]]

                start_idx = max(ii - path_len, 0)
                cur_path = [traj['x2'][start_idx:ii], traj['y2'][start_idx:ii]]

                path.set_data(cur_path)
                line.set_data(cur_x, cur_y)

            step_text.set_text(step_template.format(ii))

            return lines + paths

        anim = animation.FuncAnimation(fig, animate, range(1, n_steps),  # type: ignore
                                       interval=50, blit=True, init_func=init)
        if show_fig:
            plt.show()
        return anim

    def visualize_traj(self, n_samples2vis=10):
        """Create a plot of `n_samples2vis` randomly chosen 4D trajectories of the loaded pendulum dataset.

        Each component (th1, w1, th2, w2) is displayed in a separate subplot.

        Parameters
        ----------
        n_samples2vis : int, optional
            Controls how many trajectories are randomly chosen form the dataset. Defaults to 10.
        """
        n_samples, _, _ = self.data.shape
        chosen_idcs = np.random.choice(np.arange(n_samples), size=n_samples2vis)

        labels = [r"$\theta_1$", r"$w_1$", r"$\theta_2$", r"$w_2$"]
        _, axes = plt.subplots(4, 1, figsize=(6, 12))
        for axis_idx, ax in enumerate(axes.flat):
            for cidx in chosen_idcs:
                ax.plot(self.data[cidx, :, axis_idx])
                ax.set_xlabel('time t')
                ax.set_ylabel(labels[axis_idx])

        plt.tight_layout()
        plt.show()


class ThreeBody(BaseDataset):
    """Dataset for trajectories of a three-body problem.

    Datasets can be generated with :class:`tsfewshot.data.odeutils.ODEDatasetGenerator`.
    For details on the generating process, see :func:`tsfewshot.data.odeutils.generate_three_body_dataset`.

    * x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3 - x, y, and z position of the three bodies.
    * v_1_x, v_1_y, v_1_z, v_2_x, v_2_y, v_2_z, v_3_x, v_3_y, v_3_z - x, y, and z velocity of the three bodies.

    Parameters
    ---------
    cfg : Config
        Run configuration
    split : {'train', 'val', 'test'}
        Period for which the dataset will be used.
    dataset : str, optional
        If provided, the dataset will ignore the settings in `cfg` and use this dataset instead.
    is_train : bool, optional
        Indicates whether the dataset will be used for training or evaluation (including finetuning).
    train_scaler : Dict[str, Dict[str, torch.Tensor]], optional
        Pre-calculated scaler to use for normalization of input/output values.
    train_val_split : float, optional
        float between 0 and 1 to subset the created dataset. If provided, the created dataset will hold a dictionary
        mapping each dataset name to the indices used in the train split. Subsequently, these indices can be used
        to subset the corresponding validation datasets.
    silent : bool, optional
        Option to override cfg.silent.
    """

    def __init__(self, cfg: Config,
                 split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):
        self.data = None
        self.params = None
        super().__init__(cfg, split, dataset=dataset, is_train=is_train, train_scaler=train_scaler,
                         train_val_split=train_val_split, silent=silent)

    def _load_dataset(self, cfg: Config, dataset: str) -> Optional[xr.Dataset]:
        train_file = cfg.base_dir / f'{dataset}'
        if not train_file.exists():
            LOGGER.warning(f'Dataset {dataset} has no npy file. skipping.')
            return None

        all_data = np.load(train_file, allow_pickle=True)

        self.data = all_data['data']
        self.params = all_data['params'].item()

        if np.isnan(self.data).any():
            LOGGER.warning(f'Dataset {dataset} has NaNs. Skipping.')
            return None

        for name, (min_, max_) in cfg.filter_samples.items():
            idx = ['x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3',
                   'v_1_x', 'v_1_y', 'v_1_z', 'v_2_x', 'v_2_y', 'v_2_z',
                   'v_3_x', 'v_3_y', 'v_3_z'].index(name)
            self.data = self.data[np.all((self.data[:, :, idx] >= min_) & (self.data[:, :, idx] <= max_), axis=1)]
        if self.data.shape[0] == 0:
            LOGGER.warning(f'Dataset {dataset} has not samples. Skipping.')
            return None

        if cfg.dataset_max_trajectories > 0 and self.data.shape[0] > cfg.dataset_max_trajectories:
            LOGGER.info(f'Limiting dataset {dataset} from {self.data.shape[0]} '
                        f'to {cfg.dataset_max_trajectories} trajectories.')
            self.data = self.data[:cfg.dataset_max_trajectories]

        xarray = xr.Dataset({k: (['sample', 'step'], self.data[:, :, i])
                             for i, k in enumerate(['x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3',
                                                    'v_1_x', 'v_1_y', 'v_1_z', 'v_2_x', 'v_2_y', 'v_2_z',
                                                    'v_3_x', 'v_3_y', 'v_3_z'])})
        for param, value in self.params.items():
            xarray[param] = (('sample', 'step'), np.full(self.data.shape[:2], value))

        return xarray

    def visualize(self, nfigures_perside: int = 3, show_fig: bool = True) -> animation.FuncAnimation:
        """Create an animation with ``nfigures_perside**2`` randomly chosen three-body setups of the loaded dataset.

        To display the animation in a Jupyter notebook, use ``show_fig = False``:

        .. code-block:: python

            anim = ds.visualize(show_fig=False)
            display(IPython.display.HTML(anim.to_html5_video()))  # show the animation
            plt.clf()  # prevent matplotlib from displaying the un-animated figure

        Parameters
        ----------
        nfigures_perside : int, optional
            Controls how many problems are animated. Defaults to 3.
        show_fig : bool, optional
            If False, will not call plt.show(). Useful when called from a Jupyter notebook.

        Returns
        -------
        animation.FuncAnimation
            Animated three-body trajectories.
        """

        n_samples2vis = nfigures_perside**2
        n_samples, n_steps, _ = self.data.shape
        chosen_idcs = np.random.choice(np.arange(n_samples), size=n_samples2vis, replace=n_samples2vis >= n_samples)

        chosen_traj = []
        fig, axes = plt.subplots(nrows=nfigures_perside, ncols=nfigures_perside, figsize=(10, 10))
        for cidx, ax in zip(chosen_idcs, axes.flat):
            traj = self.data[cidx, :, :]
            x_1, y_1 = traj[:, 0], traj[:, 1]
            x_2, y_2 = traj[:, 3], traj[:, 4]
            x_3, y_3 = traj[:, 6], traj[:, 7]

            xmin = min([min(x_1), min(x_2), min(x_3)])
            xmax = max([max(x_1), max(x_2), max(x_3)])
            ymin = min([min(y_1), min(y_2), min(y_3)])
            ymax = max([max(y_1), max(y_2), max(y_3)])
            chosen_traj.append({'x_1': x_1, 'y_1': y_1, 'x_2': x_2, 'y_2': y_2, 'x_3': x_3, 'y_3': y_3})

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.grid()

        points, paths = [[], [], []], [[], [], []]
        for cidx, ax in zip(chosen_idcs, axes.flat):
            for i in range(3):
                points[i].append(ax.scatter([], [], marker='o', color=f'C{i}'))
                path, = ax.plot([], [], color=f'C{i}', alpha=0.5)
                paths[i].append(path)

        step_template = 'step = {:d}'
        step_text = fig.suptitle('')

        def init():
            for i in range(3):
                for path in paths[i]:
                    path.set_data([], [])

            step_text.set_text('')
            return points[0] + points[1] + points[2] + paths[0] + paths[1] + paths[2]

        def animate(ii, path_len=100):
            for i in range(3):
                for traj, point, path in zip(chosen_traj, points[i], paths[i]):
                    start_idx = max(ii - path_len, 0)
                    cur_path = [traj[f'x_{i+1}'][start_idx:ii], traj[f'y_{i+1}'][start_idx:ii]]

                    path.set_data(cur_path)
                    point.set_offsets(np.array([[traj[f'x_{i+1}'][ii]], [traj[f'y_{i+1}'][ii]]]).T)

            step_text.set_text(step_template.format(ii))

            return points[0] + points[1] + points[2] + paths[0] + paths[1] + paths[2]

        anim = animation.FuncAnimation(fig, animate, range(1, n_steps),  # type: ignore
                                       interval=50, blit=True, init_func=init)
        if show_fig:
            plt.show()
        return anim

    def visualize_3d(self, show_fig: bool = True) -> animation.FuncAnimation:
        """Create a 3d animation with ``nfigures_perside**2`` randomly chosen three-body setups of the loaded dataset.

        To display the animation in a Jupyter notebook, use ``show_fig = False``:

        .. code-block:: python

            anim = ds.visualize_3d(show_fig=False)
            display(IPython.display.HTML(anim.to_html5_video()))  # show the animation
            plt.clf()  # prevent matplotlib from displaying the un-animated figure

        Parameters
        ----------
        show_fig : bool, optional
            If False, will not call plt.show(). Useful when called from a Jupyter notebook.

        Returns
        -------
        animation.FuncAnimation
            Animated three-body trajectories.
        """

        n_samples, n_steps, _ = self.data.shape
        chosen_idx = np.random.choice(np.arange(n_samples), size=1)[0]

        chosen_traj = []
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        traj = self.data[chosen_idx, :, :]
        x_1, y_1, z_1 = traj[:, 0], traj[:, 1], traj[:, 2]
        x_2, y_2, z_2 = traj[:, 3], traj[:, 4], traj[:, 5]
        x_3, y_3, z_3 = traj[:, 6], traj[:, 7], traj[:, 8]

        xmin = min([min(x_1), min(x_2), min(x_3)])
        xmax = max([max(x_1), max(x_2), max(x_3)])
        ymin = min([min(y_1), min(y_2), min(y_3)])
        ymax = max([max(y_1), max(y_2), max(y_3)])
        zmin = min([min(z_1), min(z_2), min(z_3)])
        zmax = max([max(z_1), max(z_2), max(z_3)])
        chosen_traj.append({'x_1': x_1, 'y_1': y_1, 'z_1': z_1,
                            'x_2': x_2, 'y_2': y_2, 'z_2': z_2,
                            'x_3': x_3, 'y_3': y_3, 'z_3': z_3})

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 0.05))
        ax.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 0.05))
        ax.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 0.05))
        ax.grid(False)
        plt.tight_layout()

        points, paths = [[], [], []], [[], [], []]
        colors = ['#003f5c', '#bc5090', '#ffa600']
        for i in range(3):
            points[i].append(ax.plot([], [], [], marker='o', color=colors[i], markersize=20)[0])
            path, = ax.plot([], [], [], color=colors[i], alpha=0.8, lw=5)
            paths[i].append(path)

        step_template = 'step = {:d}'
        step_text = fig.suptitle('')

        def init():
            for i in range(3):
                for path in paths[i]:
                    path.set_data(np.array([]), np.array([]))
                    path.set_3d_properties([])

            step_text.set_text('')
            return points[0] + points[1] + points[2] + paths[0] + paths[1] + paths[2]

        def animate(ii, path_len=80):
            for i in range(3):
                for traj, point, path in zip(chosen_traj, points[i], paths[i]):
                    start_idx = max(ii - path_len, 0)
                    cur_path = np.array([traj[f'x_{i+1}'][start_idx:ii],
                                         traj[f'y_{i+1}'][start_idx:ii],
                                         traj[f'z_{i+1}'][start_idx:ii]])
                    path.set_data(cur_path[:2])
                    path.set_3d_properties(cur_path[-1])
                    point.set_data(np.array([traj[f'x_{i+1}'][ii],
                                             traj[f'y_{i+1}'][ii]]))
                    point.set_3d_properties(traj[f'z_{i+1}'][ii])

            step_text.set_text(step_template.format(ii))

            return points[0] + points[1] + points[2] + paths[0] + paths[1] + paths[2]

        anim = animation.FuncAnimation(fig, animate, range(1, n_steps),  # type: ignore
                                       interval=50, blit=True, init_func=init)
        if show_fig:
            plt.show()
        return anim
