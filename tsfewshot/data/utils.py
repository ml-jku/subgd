import math
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
import xarray as xr
from torch import nn
from torch.utils.data import Sampler


class PositionalEncoding(nn.Module):
    """Model that augments inputs with positional encoding, either adding or concatenating the encoding to the input.

    Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Parameters
    ----------
    encoding_dim : int
        Dimension of the positional encoding. Must be one or divisible by two.
    encoding_type : {'sum', 'cat'}
        Type of the positional encoding (add or concatenate)
    dropout : float, optional
        Dropout to apply to the result of the positional encoding. Range [0, 1).
    max_len : int, optional
        Maximum length of positional encoding. Must be larger than the largest sequence length.
    """

    def __init__(self, encoding_dim: int, encoding_type: str, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()

        if encoding_dim > 1 and encoding_dim % 2 != 0:
            raise ValueError('Encoding dim must be 1 or divisible by 2.')

        self._dropout = nn.Dropout(p=dropout)

        pos_enc = torch.zeros(1, max_len, encoding_dim)  # type: ignore
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoding_dim, 2).float() * (-math.log(max_len * 2) / encoding_dim))
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos_enc)

        if encoding_type.lower() == 'cat':
            self._concatenate = True
        elif encoding_type.lower() == 'sum':
            self._concatenate = False
        else:
            raise RuntimeError(f'Unrecognized positional encoding type: {encoding_type}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for positional encoding. Concatenates or adds positional encoding to input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data of dimension ``[batch size, sequence length, input dimension]``.

        Returns
        -------
        torch.Tensor
            Input data with positional encoding.
        """
        pos_enc = self.pos_enc[:, :x.shape[1], :].repeat(x.shape[0], 1, 1)  # type: ignore
        if self._concatenate:
            x = torch.cat([x, pos_enc], -1)

        else:
            x = x + pos_enc[:, :, :x.shape[2]]

        return self._dropout(x)


class RotateFeatures:
    """Class that augments a dataset by rotating features.

    Parameters
    ----------
    rotation_file : str
        String of pattern 'varName1#varName2#...#filePath', where filePath is a path to a pickled numpy rotation matrix.
    """

    def __init__(self, base_dir: Path, rotation_spec: str):
        self._rotate_variables = rotation_spec.split('#')[:-1]

        rotation_file = rotation_spec.split('#')[-1]
        self._rotation = pickle.load((base_dir / rotation_file).open('rb'))

        if self._rotation.shape != (len(self._rotate_variables), len(self._rotate_variables)):
            raise ValueError('Invalid rotation matrix shape.')

    def augment(self, xr_data: xr.Dataset) -> xr.Dataset:
        """Rotate variables of the dataset.

        Parameters
        ----------
        xr_data : xr.Dataset
            Dataset xarray with dimensions 'sample' and 'step'.

        Returns
        -------
        xr.Dataset
            The xarray dataset with rotated variables.
        """

        rotated = np.einsum('ij,jkt->kti', self._rotation, xr_data[self._rotate_variables].to_array())
        for i, var in enumerate(self._rotate_variables):
            xr_data[f'{var}_rotated'] = (['sample', 'step'], rotated[:, :, i])

        return xr_data


class IndexSampler(Sampler):
    """Simple PyTorch sampler through a list of indices. """

    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
