from omegaconf import DictConfig
import pickle as pkl
import torch
from typing import List, Tuple

from vsa_ogm.data.sa.base_sa_dataset import BaseSingleAgentDataset
from vsa_ogm.logging import BaseLogger

class PickelSingleAgentDataset(BaseSingleAgentDataset):
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, loggers: List[BaseLogger]) -> None:
        """
        Initialize the PickelSingleAgentDataset object.

        Args:
            config (DictConfig): The configuration object containing the
                dataset information.
            loggers (List[BaseLogger]): A list of loggers to log information.
        """
        super(PickelSingleAgentDataset, self).__init__(config, loggers)

        file_path: str = config.data.data_dir

        with open(file_path, "rb") as f:
            self.point_clouds: list = pkl.load(f)

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.point_clouds)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Return the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            Tuple: A tuple containing the data and the world size.
        """
        data_batch: dict = {
            "lidar_data": torch.from_numpy(self.point_clouds[idx][:, :2]),
            "lidar_distances": None,
            "occupancy": torch.from_numpy(self.point_clouds[idx][:, 2]),
            "max_laser_distance": None,
            "robot_poses": None
        }

        return data_batch