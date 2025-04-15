# This code was derived from the lm-polygraph repository - https://github.com/IINemo/lm-polygraph/tree/main (vashuring et al. 2025) 

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict

# Instead of importing polygraph_module_init from lm_polygraph,
# we define a dummy no-op decorator.
def polygraph_module_init(func):
    return func

class Estimator(ABC):
    """
    Abstract estimator class to estimate the uncertainty of a language model.
    """

    @polygraph_module_init
    def __init__(self, stats_dependencies: List[str], level: str):
        """
        Parameters:
            stats_dependencies (List[str]): a list of statistic names that need to be calculated.
            level (str): the level of uncertainty estimation.
                Allowed values: 'sequence', 'claim', 'token'
        """
        assert level in ["sequence", "claim", "token"]
        self.level = level
        self.stats_dependencies = stats_dependencies

    @abstractmethod
    def __str__(self):
        """
        Should return a unique name of the estimator. Include parameters that affect estimates.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates the uncertainty for each sample given input statistics.
        
        Parameters:
            stats (Dict[str, np.ndarray]): A dictionary with keys corresponding to the required statistics.
        Returns:
            np.ndarray: 1-D array of uncertainty values (higher means more uncertain).
        """
        raise NotImplementedError("Not implemented")
