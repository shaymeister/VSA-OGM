from omegaconf import DictConfig
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


