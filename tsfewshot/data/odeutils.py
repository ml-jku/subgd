# pylint: disable=invalid-name
import functools
import hashlib
import inspect
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from numpy import cos, sin
from scipy.integrate import solve_ivp
from scipy.stats import special_ortho_group
from tqdm import tqdm

sys.path.append('.')
from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset
from tsfewshot.data.odedataset import DoublePendulum, ODEDatasetSimple, ThreeBody


class ODEDatasetGenerator:
    """Generate trajectories of a system of ODEs from different initial states y0 and some parameters for the ODEs.

    Different y0 correspond to different samples, different params correspond to different tasks, and
    one instance of ODEDatasetGenerator is meant to generate samples of one task.

    Parameters
    ----------
    save_path : str
        Path to folder in which the generated dataset will be stored as a .npy file.
    name_template : str
        Template string defining how the filename for the dataset is derived from its parameters.
    n_samples : int
        Number of trajectories to generate. Should match the number of initial states.
    t0 : float
        Starting time from which the solver should generate trajectories.
    y0 : np.ndarray
        Array where initial positions are stored rowwise.
    dt : float
        Time between points where trajectories are calculated.
    n_steps : int
        number of time steps which shall be solved
    F : Callable[[float, np.ndarray], np.ndarray]
        Vector valued function returning the derivatives of the ODE system. Passed into `scipy.integrate.solve_ivp`.
        Right-hand side of the system. The calling signature is fun(t, y). Here t is a scalar, and there are two
        options for the ndarray y: It can either have shape (n,); then fun must return array_like with shape (n,).
        Alternatively, it can have shape (n, k); then fun must return an array_like with shape (n, k), i.e.,
        each column corresponds to a single column in y. The choice between the two options is determined by vectorized
        argument (see below). The vectorized implementation allows a faster approximation of the Jacobian by finite
        differences (required for stiff solvers).
    params : dict
        Additional static parameters for the ODEs. Same for all trajectories.
    filter_max : float, optional
        Skip simulations with values larger than this value (before random transformations).
    mm_trafo : np.ndarray, optional
        Matrix to apply to the final dataset via matrix multiplication.
    elem_trafo : tuple, optional
        Element-wise transformation  to apply to the final dataset. Tuple ``(trafo_type, (indices, trafo_options))``.
        If ``trafo_type`` is 'exp', ``trafo_options must be (a, b) and this function  will apply a*exp(b*x) to the
        coordinates in the list of indices ``indices``.
        If ``trafo_type`` is 'noise', ``trafo_options`` must be a (biases, std), and this function will add random noise
        with these biases (one entry per index) and std to the coordinates in the list of indices ``indices``.
    filename_supplement : str, optional
        Optional information to concatenate to the filename.
    n_jobs : int, optional
        Number of parallel jobs to use for ODE solving.
    verbose : bool, optional
        Controls printing of generation process (default False).
    """

    def __init__(self,
                 save_path: str,
                 name_template: str,
                 n_samples: int,
                 t0: float,
                 y0: np.ndarray,
                 dt: float,
                 n_steps: int,
                 F: Callable[[float, np.ndarray], np.ndarray],
                 params: dict,
                 filter_max: float = None,
                 mm_trafo: np.ndarray = None,
                 elem_trafo: tuple = None,
                 n_jobs: int = 32,
                 verbose: bool = False):

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_template = name_template

        self.n_samples = n_samples
        self.t0 = t0
        self.dt = dt
        self.y0 = y0
        self.n_steps = n_steps
        self.F = F
        self.params = params
        self.filter_max = filter_max
        self.mm_trafo = mm_trafo
        self.elem_trafo = elem_trafo
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.t = np.arange(t0, t0 + dt * n_steps, dt)

        X = []

        assert self.y0.shape[0] >= self.n_samples, (f'Cannot generate {self.n_samples} samples from '
                                                    f'{self.y0.shape[0]} initial conditions')
        if verbose:
            print('Start solving ODE trajectories by varying initial conditions')
            print('Using following function for derivative calculation:')
            print(inspect.getsource(self.F))
            print('parameters:', self.params)

        def _solve_transform(row):
            try:
                new_x = self._compute_trajectory(param_sample=self.params, y0=row)
            except RuntimeError as err:
                print(err)
                return None

            if self.filter_max is not None and (np.abs(new_x) > self.filter_max).any():
                return None

            if self.mm_trafo is not None:
                new_x = np.einsum('ij,jk->ik', self.mm_trafo, new_x)
            if self.elem_trafo is not None:
                indices = self.elem_trafo[1][0]
                if self.elem_trafo[0] == 'exp':
                    a, b = self.elem_trafo[1][1]
                    new_x[indices] = a * np.exp(b * new_x[indices])
                elif self.elem_trafo[0] == 'noise':
                    biases, std = self.elem_trafo[1][1]
                    for idx, bias in zip(indices, biases):
                        noise = np.random.normal(loc=bias, scale=std, size=new_x[idx].shape)
                        new_x[idx] += noise
                else:
                    raise ValueError('Unknown transformation')

            return new_x

        i = 0
        while len(X) < self.n_samples and i < len(self.y0):
            # start a few more than needed because some simulations will fail
            block_size = max(int(1.5 * (self.n_samples - len(X))), 4)
            y0_block = self.y0[i:i + block_size]
            traj_block = Parallel(n_jobs=self.n_jobs, verbose=1)(delayed(_solve_transform)(row) for row in y0_block)
            X += [traj for traj in traj_block if traj is not None]
            i += block_size

        X = np.array(X[:self.n_samples])
        if X.ndim > 2:
            # the following will put time steps on axis-1, rest after that
            # in special case of 3-dims the following equals np.transpose(X, axes=[0,2,1])
            axs = list(range(X.ndim))
            axs = [axs[0], axs[-1]] + axs[1:-1]
            X = np.transpose(X, axes=axs)
        assert X.shape[:2] == (self.n_samples, self.n_steps), \
            f'Unexpected result shape {X.shape} after calculation of trajectories.'

        self.X_train = X

        self._save()

    def _save(self):
        path = self.save_path / self.filename
        if path.exists():
            warnings.warn(f'{path} already exists, overwriting now')

        with path.open('wb') as f:
            if self.mm_trafo is not None or self.elem_trafo is not None or self.filter_max is not None:
                np.savez(f,
                         data=self.train_data,
                         params=self.params,
                         t0=self.t0,
                         dt=self.dt,
                         y0=self.y0,
                         mm_trafo=self.mm_trafo,
                         elem_trafo=self.elem_trafo,
                         filter_max=self.filter_max,
                         n_steps=self.n_steps)
            else:
                np.savez(f,
                         data=self.train_data,
                         params=self.params,
                         t0=self.t0,
                         dt=self.dt,
                         y0=self.y0,
                         n_steps=self.n_steps)

        if self.verbose:
            print(f'Done saving dataset to {self.save_path}')

    @property
    def train_data(self) -> np.ndarray:
        """Return the computed trajectories. """
        return self.X_train

    @property
    def filename(self) -> str:
        """Return the filename in which the dataset is saved. """
        return self.name_template.format(**self.params) + '.npy'

    def _compute_trajectory(self, param_sample, y0):
        func = functools.partial(self.F, **param_sample)
        sol = solve_ivp(func, [self.t0, self.t[-1]], y0, t_eval=self.t)

        if not sol.success:
            raise RuntimeError(sol.message)

        return sol.y.squeeze()


def generate_simple_ode_dataset(base_dir: str,
                                seed: int = 1234,
                                n_samples: int = 500,
                                t0: float = 0,
                                dt: float = 0.1,
                                n_steps: int = 200,
                                y0: np.ndarray = None,
                                params: dict = None) -> Tuple[ODEDatasetSimple, ODEDatasetGenerator]:
    r"""Generate simple ODE dataset.

    This will generate a dataset with trajectories resulting from different initial starting positions `y0`
    and evolving a dynamical system in time for `n_steps` with stepsize `dt` starting from time `t0`.
    The dynamical system ODE is given as $x'(t) = A * \cos(w*t+phi)$

    Parameters
    ----------
    base_dir : str
        Path to the folder where the dataset is stored as a .npy file
    seed : int, optional
        Random seed to control the sampling of the initial conditions.
        If one wants to continue using the current seed, set seed=None. Defaults to 1234.
    n_samples : int, optional
        Number of trajectories we want to generate and save in the dataset. Defaults to 500.
    t0 : float, optional
        Starting time from which the ODE solutions shall be calculated. Defaults to 0.
    dt : float, optional
        Size of time steps in the solution trajectories. Defaults to 0.1.
    n_steps : int, optional
        Number of time steps to solve for each trajectory. Defaults to 200.
    y0 : np.ndarray, optional
        Optionally, we can pass pre-generated initial conditions. Defaults to None.
    params : dict, optional
        Optionally, we can pass pre-defined parameters for the characteristic ODE constants 'A','w', 'phi'.
        Defaults to None.

    Returns
    -------
    Tuple[ODEDatasetSimple, ODEDatasetGenerator]
        A dataset holding the computed trajectories and an instance of the generator class holding the parameters
        which were used during generation.
    """
    if not seed is None:
        np.random.seed(seed)

    if y0 is None:
        y0 = np.array([
            np.random.rand(n_samples) * 5  # pos1
        ]).T
    else:
        assert isinstance(y0, np.ndarray), 'The initial states need to be stored in an np.ndarray'
        assert y0.ndim == 2 and y0.shape[0] == n_samples, (f'Only got {y0.shape[0]} initial conditions, '
                                                           f'but to create the dataset with {n_samples} samples '
                                                           f'we need {n_samples} ones.')

    F = lambda t, y, **params: params['A'] * np.cos(params['w'] * t + params['phi'])

    if params is None:
        params = {'A': 5, 'w': 0.5, 'phi': 0.75 * np.pi}
    else:
        assert isinstance(params, dict), 'The ODE parameters need to be provided as a dictionary'
        assert all(p in params for p in ['A', 'w', 'phi']), ('Not all parameters found in the provided'
                                                             'parameter dict. Please provide "A", "w", "phi"')

    name_template = 'A:{A}_w:{w}_phi:{phi}'
    gen = ODEDatasetGenerator(base_dir, name_template, n_samples, t0=t0, y0=y0,
                              dt=dt, n_steps=n_steps, F=F, params=params)

    # try loading ODE dataset
    config = Config({
        'base_dir': base_dir,
        'run_dir': Path('/tmp/tbd'),
        'input_vars': ['x'],
        'target_vars': {'x': [1]}
    })
    dataset = ODEDatasetSimple(config, 'train', dataset=gen.filename, train_scaler=BaseDataset.DUMMY_SCALER)

    return dataset, gen


def generate_double_pendulum_dataset(base_dir: str,
                                     seed: int = 1234,
                                     n_samples: int = 200,
                                     t0: float = 0,
                                     dt: float = 0.1,
                                     n_steps: int = 200,
                                     y0: np.ndarray = None,
                                     rand_rotate: Tuple[str, List[str]] = None,
                                     params: dict = None) -> Tuple[DoublePendulum, ODEDatasetGenerator]:
    """Generate double pendulum dataset.

    This will generate a dataset with trajectories that result from different initial starting positions `y0` and
    evolve based on a dynamical double pendulum system in time for `n_steps` with stepsize `dt` starting from time `t0`.

    Parameters
    ----------
    base_dir : str
        Path to the folder where the dataset is stored as a .npy file
    seed : int, optional
        Random seed to control the sampling of the initial conditions.
        If one wants to continue using the current seed, set seed=None. Defaults to 1234.
    n_samples : int, optional
        Number of trajectories we want to generate and save in the dataset. Defaults to 200.
    t0 : float, optional
        Starting time from which the ODE solutions shall be calculated. Defaults to 0.
    dt : float, optional
        Size of time steps in the solution trajectories. Defaults to 0.1.
    n_steps : int, optional
        Number of time steps to solve for each trajectory. Defaults to 200.
    y0 : np.ndarray, optional
        Optionally, we can pass pre-generated initial conditions. Defaults to None.
    rand_rotate : Tuple[str, List[str]], optional
        The coordinates for the pendulums provided in the list (options 'p1', 'p2') will be randomly rotated.
        The first tuple entry must be 'individual', so each vector (angle+velocity of a pendulum) is rotated by itself.
    params : dict, optional
        Optionally, we can pass pre-defined parameters for the characteristic ODE constants 'l1','l2', 'm1', 'm2'.
        Defaults to None.

    Returns
    -------
    Tuple[DoublePendulum, ODEDatasetGenerator]
        A dataset holding the computed trajectories and an instance of the generator class holding the parameters
        which were used during generation.
    """
    if not seed is None:
        np.random.seed(seed)

    def double_pendulum_deriv(th1: float,
                              w1: float,
                              th2: float,
                              w2: float,
                              l1: float = 1,
                              l2: float = 1,
                              m1: float = 1,
                              m2: float = 1,
                              g: float = 10) -> np.ndarray:
        """Calculate the derivatives of a double pendulum state.

        For the derivation of the equations, see the Jupyter notebook in ``notebooks/``.

        Parameters
        ----------
        th1 : float
            Angle (in radians) of displacement of the 1st pendulum.
        w1 : float
            Angle velocity of the 1st mass.
        th2 : float
            Angle (in radians) of displacement of the 2nd pendulum.
        w2 : float
            Angle velocity of the 2nd mass.
        l1 : float, optional
            Length of 1st pendulum arm in meters . Defaults to 1.
        l2 : float, optional
            Length of 2nd pendulum arm in meters. Defaults to 1.
        m1 : float, optional
            Mass of 1st pendulum in kilos. Defaults to 1.
        m2 : float, optional
            Mass of 2nd pendulum in kilos. Defaults to 1.
        g : float, optional
            Gravitational acceleration. Defaults to 10.

        Returns
        -------
        np.ndarray
            Time derivatives of th1, w1, th2, w2.
        """
        dth1 = w1  # dth1/dt = w1
        dth2 = w2  # dth2/dt = w2

        delta = th1 - th2
        nom1 = (dth1**2 * l1 * m2 * sin(2 * th1 - 2 * th2)) / 2 + dth2**2 * l2 * m2 * sin(delta) + \
            g * m1 * sin(th1) + (g * m2 * sin(th1)) / 2 + (g * m2 * sin(th1 - 2 * th2)) / 2
        den1 = l1 * (-m1 + m2 * cos(delta)**2 - m2)

        ddth1 = nom1 / den1

        nom2 = (m1 + m2) * (dth1**2 * l1 * sin(delta) - g * sin(th2)) + \
            (dth2**2 * l2 * m2 * sin(delta) + g * sin(th1) * (m1 + m2)) * cos(delta)
        den2 = l2 * (m1 - m2 * cos(delta)**2 + m2)
        ddth2 = nom2 / den2

        return np.array([dth1, ddth1, dth2, ddth2])

    F = lambda t, state, **params: double_pendulum_deriv(state[0], state[1], state[2], state[3], **params)

    if y0 is None:
        y0 = np.array([
            np.radians(np.random.rand(n_samples) * 180),  # th1
            np.random.randn(n_samples),  # w1
            np.radians(np.random.rand(n_samples) * 180),  # th2
            np.random.randn(n_samples)  # w2
        ]).T
    else:
        assert isinstance(y0, np.ndarray), 'The initial states need to be stored in an np.ndarray'
        assert y0.ndim == 2 and y0.shape[0] == n_samples, (f'Only got {y0.shape[0]} initial conditions, '
                                                           f'but to create the dataset with {n_samples} samples '
                                                           f'we need {n_samples} ones.')

    if params is None:
        params = {'l1': 1, 'l2': 1, 'm1': 1.5, 'm2': 1, 'g': 10}
    else:
        assert isinstance(params, dict), 'The ODE parameters need to be provided as a dictionary'
        assert all(p in params for p in ['l1', 'l2', 'm1', 'm2', 'g']), ('Not all parameters found in the provided'
                                                                         'parameter dict. Please provide '
                                                                         '"l1", "l2", "m1", "m2", "g"')

    name_template = 'L1:{l1}_L2:{l2}_M1:{m1}_M2:{m2}_g{g}'
    rotation = None
    if rand_rotate is not None:
        if rand_rotate[0].lower() == 'individual':
            # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
            alpha = np.radians(np.random.rand(1)[0] * 360)
            s_a, c_a = np.sin(alpha), np.cos(alpha)
            rotation = np.zeros((4, 4))
            for i, pendulum in zip(range(4), ['p1', 'p2']):
                if pendulum in rand_rotate[1]:
                    rotation[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = np.array([[c_a, -s_a], [s_a, c_a]])
                else:
                    rotation[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = np.eye(2)

            name_template += f'_alpha{alpha}'
        else:
            raise ValueError(f'Unknown rotation type {rand_rotate[0]}')

    gen = ODEDatasetGenerator(base_dir, name_template, n_samples, t0=t0, y0=y0,
                              mm_trafo=rotation, dt=dt, n_steps=n_steps, F=F, params=params)  # type: ignore

    config = Config({
        'base_dir': base_dir,
        'run_dir': Path('/tmp/tbd'),
        'input_vars': ['th1', 'w1', 'th2', 'w2'],
        'target_vars': {'th1': [1], 'th2': [1]}
    })
    dataset = DoublePendulum(config, 'train', dataset=gen.filename, train_scaler=BaseDataset.DUMMY_SCALER)
    return dataset, gen


def generate_multiple_double_pendulum_datasets(base_dir: str,
                                               seed: int = 1234,
                                               n_datasets: int = 50,
                                               nsamples_per_dataset: int = 200,
                                               nsteps_per_dataset: int = 200,
                                               rand_rotate: Tuple[str, List[str]] = None,
                                               parameters: dict = None,
                                               **kwargs) -> Dict[str, Tuple[DoublePendulum, ODEDatasetGenerator]]:
    """Generate multiple double pendulum datasets.

    Parameters
    ----------
    base_dir : str
        Path to the folder where the datasets are stored as .npy files.
    seed : int, optional
        Random seed to control the sampling of the initial conditions.
        If one wants to continue using the current seed, set seed=None. Defaults to 1234.
    n_datasets : int, optional
        Number of datasets to generate. Defaults to 50.
    nsamples_per_dataset : int, optional
        Number of trajectories to generate and save in each dataset. Defaults to 200.
    nsteps_per_dataset : int, optional
        Number of time steps to solve for each trajectory in each dataset. Defaults to 200.
    rand_rotate : Tuple[str, List[str]], optional
        The coordinates for the pendulums provided in the list (options 'p1', 'p2') will be randomly rotated.
        The first tuple entry must be 'individual', so each vector (angle+velocity of a pendulum) is rotated by itself.
    parameters : dict, optional
        Optionally, we can pass pre-defined parameters for the characteristic ODE constants 'l1','l2', 'm1', 'm2', 'g'.
        Defaults to None.

    Returns
    -------
    Dict[str, Tuple[DoublePendulum, ODEDatasetGenerator]]
        A dictionary of all datasets, each holding the computed trajectories and an instance of the generator class
        holding the parameters which were used during generation.
    """
    if not seed is None:
        np.random.seed(seed)

    if parameters is None:
        # [0, 1) * -1 + 1 -> (0,1]. Since lengths and masses close to zero create systems wich are hard to solve,
        # we use [0,1) + 0.5 -> [0.5, 1.5).
        parameters = {
            'l1': np.random.rand(n_datasets) + 0.5,
            'l2': np.random.rand(n_datasets) + 0.5,
            'm1': np.random.rand(n_datasets) + 0.5,
            'm2': np.random.rand(n_datasets) + 0.5,
            'g': np.ones(n_datasets) * 10
        }
    else:
        assert isinstance(parameters, dict), 'The ODE parameters need to be provided as a dictionary'
        assert all(p in parameters for p in ['l1', 'l2', 'm1', 'm2', 'g']), ('Not all parameters found in the provided'
                                                                             'parameter dict. Please provide '
                                                                             '"l1", "l2", "m1", "m2", "g".')
        assert all(val.shape == (n_datasets,) for val in parameters.values()), ('Each provided parameter needs to have '
                                                                                f'shape ({n_datasets},) inside the '
                                                                                'parameters dict')

    result = dict()
    print('Generating pendulum datasets')
    for idx in tqdm(range(n_datasets)):
        param = {k: v[idx] for k, v in parameters.items()}

        try:
            dataset, gen = generate_double_pendulum_dataset(base_dir, seed=None,  # type: ignore
                                                            n_samples=nsamples_per_dataset,
                                                            n_steps=nsteps_per_dataset,
                                                            rand_rotate=rand_rotate,
                                                            params=param, **kwargs)
        except RuntimeError as err:
            print(f'Error generating dataset with parameters {param}')
            print(err)
            continue
        if gen.filename in result:
            warnings.warn(f'Dataset {gen.filename} was already generated!')
        else:
            result[gen.filename] = (dataset, gen)

    assert len(result) == n_datasets, 'Failed to generate some datasets or parameters were the same. Use different seed'
    return result


def generate_three_body_dataset(base_dir: str,
                                seed: int = 1234,
                                n_samples: int = 200,
                                t0: float = 0,
                                dt: float = 0.1,
                                n_steps: int = 200,
                                y0: np.ndarray = None,
                                filter_max: float = None,
                                rand_rotate: Tuple[str, List[str]] = None,
                                rand_trafo: Tuple[str, List[str]] = None,
                                name_suffix: str = None,
                                params: dict = None) -> Tuple[ThreeBody, ODEDatasetGenerator]:
    """Generate three-body dataset.

    This will generate a dataset with trajectories that result from different initial starting positions `y0` and
    evolve based on a three-body system in time for `n_steps` with stepsize `dt` starting from time `t0`.

    Parameters
    ----------
    base_dir : str
        Path to the folder where the dataset is stored as a .npy file
    seed : int, optional
        Random seed to control the sampling of the initial conditions.
        If one wants to continue using the current seed, set seed=None. Defaults to 1234.
    n_samples : int, optional
        Number of trajectories we want to generate and save in the dataset. Defaults to 200.
    t0 : float, optional
        Starting time from which the ODE solutions shall be calculated. Defaults to 0.
    dt : float, optional
        Size of time steps in the solution trajectories. Defaults to 0.1.
    n_steps : int, optional
        Number of time steps to solve for each trajectory. Defaults to 200.
    y0 : np.ndarray, optional
        Optionally, we can pass pre-generated initial conditions. Defaults to None.
    filter_max : float, optional
        Skip simulations with values larger than this value.
    rand_rotate : Tuple[str, List[str]], optional
        The coordinates for the masses provided in the list (options 'm1', 'm2', 'm3') will be randomly rotated.
        If the first tuple entry is 'individual', each vector (position/velocity of a mass) is rotated by itself.
        If the first tuple entry is 'combined', the concatenated vector of positions and velocities of all masses
        provided in the list is rotated. Note that the latter option has no physical interpretation but allows to
        generate rotated inputs that do not correspond to just another (un-rotated) initial condition.
    rand_trafo : Tuple[str, List[str]], optional
        If the str is 'exp', the coordinates for the masses provided in the list (options 'm1', 'm2', 'm3') will be
        randomly transformed element-wise as a*exp(b*x) with random a, b.
        If the str is 'noise', the coordinates for the masses provided in the list (options 'm1', 'm2', 'm3') will be
        disturbed with Gaussian noise that has random biases.
    name_suffix : str, optional
        Suffix to append to the dataset name.
    params : dict, optional
        Optionally, we can pass pre-defined parameters for the characteristic ODE constants 'm1', 'm2', 'm3'.
        Defaults to None.

    Returns
    -------
    Tuple[ThreeBody, ODEDatasetGenerator]
        A dataset holding the computed trajectories and an instance of the generator class holding the parameters
        which were used during generation.
    """
    if not seed is None:
        np.random.seed(seed)

    def three_body_deriv(x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3,
                         v_1_x, v_1_y, v_1_z, v_2_x, v_2_y, v_2_z, v_3_x, v_3_y, v_3_z,
                         m1: float, m2: float, m3: float,
                         g: float = 1) -> np.ndarray:
        """Calculate the derivatives of a three-body state.

        Parameters
        ----------
        x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3 : float
            x, y, and z position of the three bodies.
        v_1_x, v_1_y, v_1_z, v_2_x, v_2_y, v_2_z, v_3_x, v_3_y, v_3_z : float
            x, y, and z velocity of the three bodies.
        m1, m2, m3: float
            Masses of the three bodies.
        g : float, optional
            Gravitational constant. Defaults to 1.

        Returns
        -------
        np.ndarray
            Time derivatives of x_1, ..., z_3, v_1_x, ..., v_3_z.
        """
        pos_1 = np.array([x_1, y_1, z_1])
        pos_2 = np.array([x_2, y_2, z_2])
        pos_3 = np.array([x_3, y_3, z_3])

        dv_1 = -g * m2 * (pos_1 - pos_2) / (np.linalg.norm(pos_1 - pos_2)**3) \
            - g * m3 * (pos_1 - pos_3) / (np.linalg.norm(pos_1 - pos_3)**3)
        dv_2 = -g * m3 * (pos_2 - pos_3) / (np.linalg.norm(pos_2 - pos_3)**3) \
            - g * m1 * (pos_2 - pos_1) / (np.linalg.norm(pos_2 - pos_1)**3)
        dv_3 = -g * m1 * (pos_3 - pos_1) / (np.linalg.norm(pos_3 - pos_1)**3) \
            - g * m2 * (pos_3 - pos_2) / (np.linalg.norm(pos_3 - pos_2)**3)

        # dx_1 is v_1_x
        return np.array([v_1_x, v_1_y, v_1_z, v_2_x, v_2_y, v_2_z, v_3_x, v_3_y, v_3_z,
                         dv_1[0], dv_1[1], dv_1[2], dv_2[0], dv_2[1], dv_2[2], dv_3[0], dv_3[1], dv_3[2]])

    F = lambda t, state, **params: three_body_deriv(*state, **params)

    if y0 is None:
        buffer = 20
        # generate buffer times as many initial conditions because some will not be numerically stable and fail
        # and some will generate values > filter_max.
        y0 = np.random.rand(buffer * n_samples, 9) * 10 - 5  # position
        y0 = np.concatenate([y0, np.random.rand(buffer * n_samples, 9) - 0.5], axis=1)  # velocity
    else:
        assert isinstance(y0, np.ndarray), 'The initial states need to be stored in an np.ndarray'
        assert y0.ndim == 2 and y0.shape[0] >= n_samples, (f'Only got {y0.shape[0]} initial conditions, '
                                                           f'but to create the dataset with {n_samples} samples '
                                                           f'we need at least {n_samples} ones.')

    if params is None:
        params = {'m1': 10, 'm2': 20, 'm3': 30}
    else:
        assert isinstance(params, dict), 'The ODE parameters need to be provided as a dictionary'
        assert all(p in params for p in ['m1', 'm2', 'm3']), ('Not all parameters found in the provided'
                                                              'parameter dict. Please provide '
                                                              '"m1", "m2", "m3"')

    name_template = 'M1:{m1}_M2:{m2}_M3:{m3}_g:{g}'
    rotation = None
    if rand_rotate is not None:
        if rand_rotate[0].lower() == 'individual':
            # https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
            alpha, beta, gamma = np.radians(np.random.rand(3) * 360)
            s_a, c_a = np.sin(alpha), np.cos(alpha)
            s_b, c_b = np.sin(beta), np.cos(beta)
            s_g, c_g = np.sin(gamma), np.cos(gamma)
            rotation = np.zeros((18, 18))
            for i, body in zip(range(6), ['m1', 'm2', 'm3', 'm1', 'm2', 'm3']):
                if body in rand_rotate[1]:
                    rotation[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = \
                        np.array([[c_a * c_b, c_a * s_b * s_g - s_a * c_g, c_a * s_b * c_g + s_a * s_g],
                                  [s_a * c_b, s_a * s_b * s_g + c_a * c_g, s_a * s_b * c_g - c_a * s_g],
                                  [-s_b, c_b * s_g, c_b * c_g]])
                else:
                    rotation[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = np.eye(3)

            name_template += f'_a{alpha}_b{beta}_g{gamma}'
        elif rand_rotate[0] == 'combined':

            # find the indices of the coordinates that should be rotated.
            # Each mass has 6 values (3d position + 3d velocity).
            rotation_coords = []
            for i, body in zip(range(6), ['m1', 'm2', 'm3', 'm1', 'm2', 'm3']):
                if body in rand_rotate[1]:
                    rotation_coords += list(range(i * 3, (i + 1) * 3))

            # generate rotation for the desired coordinates
            rot_matrix = special_ortho_group.rvs(len(rotation_coords))

            # merge rotation matrix for desired coordinates with identity matrix for remaining coordinates
            rotation = np.eye(18)
            for i, i_idx in enumerate(rotation_coords):
                for j, j_idx in enumerate(rotation_coords):
                    rotation[i_idx, j_idx] = rot_matrix[i, j]

            name_template += f'_r-{"".join(rand_rotate[1])}'
        else:
            raise ValueError(f'Unknown rotation type {rand_rotate[0]}')

    if rand_trafo is not None:
        indices = []
        for i, body in zip(range(6), ['m1', 'm2', 'm3', 'm1', 'm2', 'm3']):
            if body in rand_trafo[1]:
                indices += list(range(i * 3, (i + 1) * 3))
        if rand_trafo[0] == 'exp':
            a = np.random.rand(1)[0] * 8 - 4
            b = np.random.rand(1)[0] * 0.1 + 0.1
            trafo = (rand_trafo[0], (indices, (a, b)))
            name_template += f'_a{a}_b{b}'
        elif rand_trafo[0] == 'noise':
            biases = np.random.rand(len(indices)) * 20 - 10
            std = 0.25
            trafo = (rand_trafo[0], (indices, (biases, std)))
            name_template += f'_bias{hashlib.md5(str(biases).encode("utf-8")).hexdigest()}_std{std}'
        else:
            raise ValueError('Unknown transformation')
    else:
        trafo = None

    if name_suffix is not None:
        name_template += name_suffix

    gen = ODEDatasetGenerator(base_dir, name_template, n_samples, t0=t0, y0=y0,
                              dt=dt, n_steps=n_steps, F=F, params=params, filter_max=filter_max,
                              mm_trafo=rotation, elem_trafo=trafo)

    config = Config({
        'base_dir': base_dir,
        'meta_dataset': 'three_body',
        'run_dir': Path('/tmp/tbd'),
        'query_size': 1, 'support_size': 0,
        'input_vars': ['x_1', 'y_1'],
        'target_vars': {'x_3': [1], 'y_3': [1]}
    })
    dataset = ThreeBody(config, 'train', train_scaler=BaseDataset.DUMMY_SCALER, dataset=gen.filename)
    return dataset, gen


def generate_multiple_three_body_datasets(base_dir: str,
                                          seed: int = 1234,
                                          n_datasets: int = 50,
                                          nsamples_per_dataset: int = 80,
                                          nsteps_per_dataset: int = 100,
                                          parameters: dict = None,
                                          filter_max: float = None,
                                          rand_rotate: Tuple[str, List[str]] = None,
                                          rand_trafo: Tuple[str, List[str]] = None,
                                          add_suffix: bool = False,
                                          **kwargs) -> Dict[str, Tuple[ThreeBody, ODEDatasetGenerator]]:
    """Generate multiple three-body datasets.

    Parameters
    ----------
    base_dir : str
        Path to the folder where the datasets are stored as .npy files.
    seed : int, optional
        Random seed to control the sampling of the initial conditions.
        If one wants to continue using the current seed, set seed=None. Defaults to 1234.
    n_datasets : int, optional
        Number of datasets to generate. Defaults to 50.
    nsamples_per_dataset : int, optional
        Number of trajectories to generate and save in each dataset. Defaults to 200.
    nsteps_per_dataset : int, optional
        Number of time steps to solve for each trajectory in each dataset. Defaults to 200.
    parameters : dict, optional
        Optionally, we can pass pre-defined parameters for the characteristic ODE constants 'm1', 'm2', 'm3'.
        Defaults to None.
    filter_max : float, optional
        Skip simulations with values larger than this value.
    rand_rotate : Tuple[str, List[str]], optional
        The coordinates for the masses provided in the list (options 'm1', 'm2', 'm3') will be randomly rotated.
        If the first tuple entry is 'individual', each vector (position/velocity of a mass) is rotated by itself.
        If the first tuple entry is 'combined', the concatenated vector of positions and velocities of all masses
        provided in the list is rotated. Note that the latter option has no physical interpretation but allows to
        generate rotated inputs that do not correspond to just another (un-rotated) initial condition.
    rand_trafo : Tuple[str, List[str]], optional
        If the str is 'exp', the coordinates for the masses provided in the list (options 'm1', 'm2', 'm3') will be
        randomly transformed element-wise as a*exp(b*x) with random a, b.
        If the str is 'noise', the coordinates for the masses provided in the list (options 'm1', 'm2', 'm3') will be
        disturbed with Gaussian noise that has random biases.
    add_suffix : bool, optional
        If True, the names of the generated datasets will be suffixed with an increasing index to prevent overwriting
        the datasets previously generated during this method call.

    Returns
    -------
    Dict[str, Tuple[ThreeBody, ODEDatasetGenerator]]
        A dictionary of all datasets, each holding the computed trajectories and an instance of the generator class
        holding the parameters which were used during generation.
    """
    if not seed is None:
        np.random.seed(seed)

    if parameters is None:
        parameters = {
            'm1': np.random.rand(n_datasets) * 50 + 2,
            'm2': np.random.rand(n_datasets) * 50 + 2,
            'm3': np.random.rand(n_datasets) * 50 + 2,
        }
    else:
        assert isinstance(parameters, dict), 'The ODE parameters need to be provided as a dictionary'
        assert all(p in parameters for p in ['m1', 'm2', 'm3']), ('Not all parameters found in the provided'
                                                                  'parameter dict.')
        assert all(val.shape == (n_datasets,) for val in parameters.values()), ('Each provided parameter needs to have '
                                                                                f'shape ({n_datasets},) inside the '
                                                                                'parameters dict')

    result = dict()
    print('Generating three-body datasets')
    for idx in tqdm(range(n_datasets)):
        param = {k: v[idx] for k, v in parameters.items()}

        name_suffix = None
        if add_suffix:
            name_suffix = f'_{idx}'

        try:
            dataset, gen = generate_three_body_dataset(base_dir, seed=None,  # type: ignore
                                                       n_samples=nsamples_per_dataset,
                                                       n_steps=nsteps_per_dataset, params=param,
                                                       filter_max=filter_max,
                                                       rand_rotate=rand_rotate,
                                                       rand_trafo=rand_trafo,
                                                       name_suffix=name_suffix,
                                                       **kwargs)
        except RuntimeError as err:
            print(f'Error generating dataset with parameters {param}')
            print(err)
            continue
        if gen.filename in result:
            warnings.warn(f'Dataset {gen.filename} was already generated!')
        else:
            result[gen.filename] = (dataset, gen)

    if len(result) != n_datasets:
        print('Failed to generate some datasets or parameters were the same. Use different seed')
    return result


if __name__ == '__main__':
    MODE = 'create_three_body'

    ##########################################################
    # try generating simple ODE dataset + saving
    ##########################################################
    if MODE == 'test_simple':  # type: ignore
        ds, _ = generate_simple_ode_dataset('/tmp/ODE_DATA')
        ds.visualize(n_samples2vis=20)

    ##########################################################
    # try generating one double pendulum dataset with fixed lengths/weights
    ##########################################################
    if MODE == 'test_pendulum':  # type: ignore
        ds, _ = generate_double_pendulum_dataset('/tmp/ODE_DATA')
        ds.visualize(nfigures_perside=3)
        ds.visualize_traj(n_samples2vis=10)

    ##########################################################
    # create double pendulum dataset with different pendulum lengths/weights
    ##########################################################
    if MODE == 'create_pendulum':  # type: ignore
        n = 120
        pendulums = {'l1': np.ones(n), 'l2': np.ones(n), 'm1': np.ones(n), 'm2': np.ones(n), 'g': np.ones(n) * 9.8}
        pendulums = {
            'l1': np.random.rand(n) + 0.5,
            'l2': np.random.rand(n) + 0.5,
            'm1': np.random.rand(n) + 0.5,
            'm2': np.random.rand(n) + 0.5,
            'g': np.random.rand(n) * 20 - 10
        }
        all_datasets = generate_multiple_double_pendulum_datasets('/tmp/ODE_DATA', n_datasets=n,
                                                                  nsamples_per_dataset=100,
                                                                  nsteps_per_dataset=120,
                                                                  parameters=pendulums)
        for ds, _ in all_datasets.values():
            ds.visualize(nfigures_perside=3)
            ds.visualize_traj(n_samples2vis=10)

    if MODE == 'create_three_body':  # type: ignore
        n = 200
        masses = {'m1': np.ones(n) * 20, 'm2': np.ones(n) * 40, 'm3': np.ones(n) * 30}

        np.random.seed(0)
        masses['g'] = np.random.rand(n) * 2
        all_datasets = generate_multiple_three_body_datasets('/tmp/ODE_DATA', n_datasets=n,
                                                             nsamples_per_dataset=5000,
                                                             #rand_trafo=('noise', ['m1', 'm2']),
                                                             #rand_rotate=('individual', ['m1']),
                                                             parameters=masses,
                                                             seed=2,
                                                             nsteps_per_dataset=120)
        for ds, _ in all_datasets.values():
            an = ds.visualize()
