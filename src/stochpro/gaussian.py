"""Gaussian Process.
"""

import numpy as np

from stochpro.base import RandomProcess
from stochpro.utils import generate_times

from typing import Iterable
from typing import Callable


class GaussianProcess(RandomProcess):
    """Gaussian process.

    A stochastic process :math:`X = (X_t)_{t \\geq 0}` is a
    **Gaussian process** if for all :math:`n \\in \\mathbb{N}` and all
    time sequences :math:`t_1,\\ldots,t_n \\geq 0` with
    :math:`0 < t_1 < \\ldots < t_n` applies

    .. math::

        \\left( X_{t_1}, \\ldots, X_{t_n} \\right) \\sim \\mathcal{N}
        (m, \\Sigma)

    with

    + :math:`m` a vector of expectations and
    + :math:`\\Sigma` a covariance matrix,

    i.e.
    :math:`\\left( X_{t_1}, \\ldots, X_{t_n} \\right)` follows a
    :math:`n`-dimensional normal distribution.

    Notes
    -----
    For more information, see
    `Gaussian process <https://en.wikipedia.org/wiki/Gaussian_process>`_.

    Parameters
    ----------
    expectation_function : callable
        The expectation function :math:`m(t) = E(X_t)`.
    covariance_function : callable
        The covariance function :math:`\\sigma (s, t) = Cov(X_s, X_t)`.
    t : float, optional
        The right hand side of the time interval :math:`[0,t]`, by
        default 1.0.

    Methods
    -------
    sample
    sample_at
    """

    def __init__(
        self,
        expectation_function: Callable[[float], float],
        covariance_function: Callable[[float, float], float],
        t: float = 1
    ) -> None:
        self.exp_func = expectation_function
        self.cov_func = covariance_function
        super().__init__(t)

    def sample(self, n: int) -> np.array:
        """Generate a sample of a Gaussian process.

        Generate a realization with :math:`n` increments.

        Parameters
        ----------
        n : int
            Number of increments.

        Returns
        -------
        np.array
            A sample of the Gaussian process.
        """
        delta_t = 1.0 * self.t / n

        gp_sample = self.rng.normal(scale=np.sqrt(delta_t), size=n)

        times = generate_times(end=self.t, n=n)

        return gp_sample

    def sample_at(self, times: Iterable[float]) -> np.array:
        """Generate a sample of a Gaussian process at specified times
        from zero.

        Parameters
        ----------
        times : array-like
            A strictly increasing time vector
            :math:`(t_1, \\ldots, t_n)`.

        Returns
        -------
        np.array
            A sample of the Gaussian process at
            :math:`(t_1, \\ldots, t_n)`.
        """
        return np.zeros(1)
