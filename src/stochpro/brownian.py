import numpy as np
from stochpro.base import RandomProcess

from typing import Iterable
from typing import Callable


class BrownianMotion(RandomProcess):
    """Brownian motion.

    Parameters
    ----------
    RandomProcess : _type_
        _description_

    Methods
    -------
    sample
    sample_at
    """

    def __init__(self, drift: Callable[[float], float], t: float = 1) -> None:
        pass

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
