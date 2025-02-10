from omegaconf import DictConfig
from typing import List, Tuple

from vsa_ogm.logging import BaseLogger

def load_data(config: DictConfig, loggers: List[BaseLogger]) -> tuple:
    """
    Load the data based on the provided configuration.

    Args:
        config (DictConfig): The configuration object containing the
            dataset information.

    Returns:
        tuple: A tuple containing the dataloader object and the world size.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    
    print(config.data)

