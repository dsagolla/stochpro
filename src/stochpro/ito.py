from stochpro.base import RandomProcess

import numpy as np

from typing import Iterable
from typing import Callable


class ItoProcess(RandomProcess):
    """Itô process.

    Parameters
    ----------
    mu : float
        _description_
    f : callable
        _description_
    g : callable
        _description_
    t : float, optional
        _description_m default is 1.0.

    Methods
    -------
    sample
    sample_at
    """

    def __init__(
        self,
        mu: float,
        f: Callable[[float], float],
        g: Callable[[float], float],
        t: float = 1
    ) -> None:
        super().__init__(t)

    def sample(self, n: int) -> np.array:
        """_summary_

        Parameters
        ----------
        n : int
            _description_

        Returns
        -------
        np.array
            _description_
        """
        pass

    def sample_at(self, times: Iterable[float]) -> np.array:
        """_summary_

        Parameters
        ----------
        times : Iterable[float]
            _description_

        Returns
        -------
        np.array
            _description_
        """
        pass
