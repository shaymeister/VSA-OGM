from omegaconf import DictConfig
from typing import List, Tuple

from vsa_ogm.data.single_agent.base_single_agent_dataset import BaseSingleAgentDataset
from vsa_ogm.logging import BaseLogger

class ToySimSingleAgentDataset(BaseSingleAgentDataset):
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, loggers: List[BaseLogger]) -> None:
        """
        Initialize the ToySimDataset object.

        Args:
            config (DictConfig): The configuration object containing the
                dataset information.
            loggers (List[BaseLogger]): A list of loggers to log information.
        """
        super(ToySimSingleAgentDataset, self).__init__(config, loggers)

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        raise NotImplementedError("The __len__ method must be implemented in the derived class.")

    def __getitem__(self, idx: int) -> Tuple:
        """
        Return the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            Tuple: A tuple containing the data and the world size.
        """
        raise NotImplementedError("The __getitem__ method must be implemented in the derived class.")