from omegaconf import DictConfig
from typing import List

from .base_sa_mapper import BaseSingleAgentMapper
from ...logging import BaseLogger

class SA_VSA_OGM(BaseSingleAgentMapper):
    """
    TODO Finish Documentation
    """
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
