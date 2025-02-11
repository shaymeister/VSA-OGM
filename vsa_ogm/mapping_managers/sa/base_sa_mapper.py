from omegaconf import DictConfig
from typing import List

from ...logging import BaseLogger

class BaseSingleAgentMapper:
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, loggers: List[BaseLogger],
                 print_header: str = "(Base SA Mapper)") -> None:
        """
        Initialize the BaseSingleAgentMapper object.

        Args:
            config (DictConfig): The configuration object containing the
                mapping information
            loggers (List[BaseLogger]): A list of loggers to log information.
            print_header (str): The header to print when logging information

        Returns:
            None
        """
        self.config: DictConfig = config
        self.loggers: List[BaseLogger] = loggers
        self.print_header: str = print_header