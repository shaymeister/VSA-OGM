import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
from typing import List

from .base_sa_mapper import BaseSingleAgentMapper
from ...logging import BaseLogger



class SA_VSA_OGM(BaseSingleAgentMapper):
    """
    TODO Finish Documentation
    """
    # class variables and objects
    num_observations: int = 0
    pairwaise_distance = nn.PairwiseDistance()


    def __init__(self, config: DictConfig, loggers: List[BaseLogger],
                 print_header: str = ("(SA VSA-OGM")) -> None:
        """
        Initialize the VSA_OGM object.

        Args:
            config (DictConfig): The configuration object containing the
                mapping information
            loggers (List[BaseLogger]): A list of loggers to log information.
            print_header (str): The header to print when logging information

        Returns:
            None
        """
        super(SA_VSA_OGM, self).__init__(config, loggers, print_header)

        # -----------------------------------------------
        # extract parameters from the config
        # -----------------------------------------------
        self.axis_resolution: int = config.mapping.axis_resolution
        self.device: str = config.mapping.device
        self.num_tiles: int = config.mapping.num_tiles
        self.seed: int = config.mapping.seed
        self.vector_dimensionality: int = config.mapping.vector_dimensionality
        self.vector_length_scale: float = config.mapping.vector_length_scale
        self.world_bounds: List[int] = config.data.world_bounds

        # -----------------------------------------------
        # initialize class variables based on the config
        # -----------------------------------------------
        
        # normalize the world bounds to start at (0, 0)
        self.world_bounds_norm: List[int] = [
            self.world_bounds[1] - self.world_bounds[0],
            self.world_bounds[3] - self.world_bounds[2]
        ]

    def fit(self, X: List[np.ndarray], y: List[np.ndarray]) -> None:
        """
        Fit the VSA_OGM model to the given data.

        Args:
            X (List[np.ndarray]): The input data to fit the model to.
            y (List[np.ndarray]): The target data to fit the model to.

        Returns:
            None
        """
        fit_metrics: dict = {}

        if len(X) != len(y):
            raise ValueError("The number of input and target data must match.")
        
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
            y = torch.tensor(y)
        
        X = X.to(self.device)
        y = y.to(self.device)

        return fit_metrics
    
    def predict(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict the output from the input data.

        Args:
            X (List[np.ndarray]): The input data to predict the output from.

        Returns:
            List[np.ndarray]: The predicted output data.
        """
        predictions: List[np.ndarray] = []
        prediction_metrics: dict = {}

        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
        
        X = X.to(self.device)

        return predictions, prediction_metrics

