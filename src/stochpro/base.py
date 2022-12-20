"""Base class.
"""
import numpy as np

from typing import Iterable

from abc import ABC
from abc import abstractmethod

from stochpro.utils import generate_times


class RandomProcess(ABC):
    """Base class for a random time processes to be subclassed by all
    processes.

    Parameters
    ----------
    t : float, optional
        The right hand side of the time interval :math:`[0,t)`, by
        default 1.0.

    Methods
    -------
    sample
    sample_at
    """

    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, t: float = 1.0) -> None:
        self.t = t

    @property
    def t(self) -> float:
        return self._t

    @t.setter
    def t(self, value: float) -> None:
        if value < 0:
            raise ValueError(
                'The end of the interval [0, t] must be positive.'
            )
        if not isinstance(value, float):
            raise TypeError(
                'The end of the interval [0, t] must be a float.'
            )
        self._t = value

    def _times(self, n: int) -> np.array:
        """Generate times associated with n increments on [0, t].

        Parameters
        ----------
        n : int
            The number of increments.

        Returns
        -------
        np.array
            A vector of times.
        """
        return generate_times(n)

    @abstractmethod
    def sample(self, n: int) -> np.array:
        pass

    @abstractmethod
    def sample_at(self, times: Iterable[float]) -> np.array:
        pass
