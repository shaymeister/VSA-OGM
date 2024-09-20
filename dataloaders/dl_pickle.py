import copy
import numpy as np
from omegaconf import DictConfig
import pickle as pkl

class PickleDataLoader:
    """
    A data loader class for loading point cloud data from a pickle file.

    Args:
        config (DictConfig): The configuration for the data loader.

    Attributes:
        env_time_step (int): The current time step of the environment.
        point_clouds (list): The list of point clouds loaded from the file.

    Raises:
        ValueError: If the environment has reached the end of the data.

    """

    def __init__(self, config: DictConfig) -> None:
        """
        Initializes a new instance of the PickleDataLoader class.

        Args:
            config (DictConfig): The configuration for the data loader.
        """

        # -----------------------------
        # TODO Add Argument Validation
        # -----------------------------

        file_path: str = config.data_dir

        self.env_time_step: int = -1

        with open(file_path, "rb") as f:
            self.point_clouds: list = pkl.load(f)

    def step(self) -> np.ndarray:
        """
        Advances the environment to the next time step and returns the
        corresponding point cloud.

        Returns:
            np.ndarray: The point cloud at the current time step.

        Raises:
            ValueError: If the environment has reached the end of the data.
        """
        self.env_time_step += 1

        if self.env_time_step >= len(self.point_clouds):
            raise ValueError("Environment has reached the end of the data.")


        pc_dict = {
            "lidar_data": self.point_clouds[self.env_time_step][:, :2],
            "occupancy": self.point_clouds[self.env_time_step][:, 2]
        }

        return copy.deepcopy(pc_dict)
    
    def max_steps(self) -> int:
        """
        Returns the maximum number of time steps in the dataset.

        Returns:
            int: The maximum number of time steps in the dataset.
        """
        return len(self.point_clouds)
    
    def reset(self) -> np.ndarray:
        """
        Resets the environment to its initial state and returns the first
        point cloud.

        Returns:
            np.ndarray: The first point cloud.
        """
        self.env_time_step = 0

        pc_dict = {
            "lidar_data": self.point_clouds[self.env_time_step][:, :2],
            "occupancy": self.point_clouds[self.env_time_step][:, 2]
        }

        return copy.deepcopy(pc_dict) 
