"""Utility functions.
"""
from typing import Iterable

import numpy as np


def generate_times(end: float, n: int) -> np.array:
    """Generate times.

    Generate a linspace from 0 to ``end`` for ``n`` increments. That is
    divide the time interval :math:`[0, end]` in :math:`n` equidistant
    intervals.

    Parameters
    ----------
    n : int
        The number of increments.
    end : float
        The right hand side of the interval :math:`[0,end]`.

    Returns
    -------
    np.array
        A vector of times.
    """
    return np.linspace(0, end, n+1)


def check_times(times: Iterable[float]) -> None:
    """Check times.

    Check if the times vector is a sequence of strictly increasing and
    nonnegative floats.

    Parameters
    ----------
    times : Iterable[float]
        A vector of times.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        Times are not a vector of floats.
    ValueError
        Times are not nonnegative or not strictly increasing.
    """

    if np.any([[not isinstance(t, float)] for t in times]):
        raise TypeError('Times must be an iterable of floats.')

    if np.any([t < 0 for t in times]):
        raise ValueError('Times must be nonnegative.')

    increments = np.diff(times)

    if np.any([t < 0 for t in increments]):
        raise ValueError('Times must be strictly increasing.')

    return None
