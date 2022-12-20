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
        The right hand side of the time interval [0, t], by default 1.0.

    Methods
    -------
    sample
    sample_at

    Raises
    ------
    TypeError
        The end of the interval [0, t] is no float.
    ValueError
        The end of the interval [0, t] is not positive.
    """

    _rng: np.random.Generator = np.random.default_rng()

    def __init__(self, t: float = 1.0) -> None:
        if not isinstance(t, float):
            raise TypeError('The end of the interval must be a float.')
        if t <= 0:
            raise ValueError('The end of the interval must be positive.')
        self._t = t

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

        Raises
        ------
        TypeError
            Number of increments is not an integer.
        """
        if not isinstance(n, int):
            raise TypeError('Number of increments must be an integer.')

        return generate_times(end=self._t, n=n)

    @abstractmethod
    def sample(self, n: int) -> np.array:
        pass

    @abstractmethod
    def sample_at(self, times: Iterable[float]) -> np.array:
        pass
