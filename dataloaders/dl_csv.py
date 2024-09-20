import copy
import numpy as np
from omegaconf import DictConfig
import pandas as pd

class CSVDataLoader():
    """
    A class representing an environment with data stored in a CSV file.

    Attributes:
        data (pd.DataFrame): The data stored in the CSV file.
        num_time_steps (int): The number of unique timestamps in the data.
        point_clouds (list): A list of numpy arrays representing the point clouds for each time step.
        env_time_step (int): The current time step of the environment.

    Methods:
        __init__(self, config: DictConfig) -> None:
            Initializes a CsvData object with the given configuration.
        step(self) -> np.ndarray:
            Returns the next point cloud in the dataset.
        max_steps(self) -> int:
            Returns the maximum number of time steps in the dataset.
        reset(self) -> np.ndarray:
            Resets the environment to its initial state and returns the first point cloud.
    """
    def __init__(self, config: DictConfig) -> None:
        """
        Initialize a CsvData object with the given configuration.

        Args:
            config (DictConfig): A configuration object containing the following keys:
                - file_path (str): The path to the CSV file containing the data.

        Returns:
            None
        """

        # -----------------------------
        # TODO Add Argument Validation
        # -----------------------------

        file_path: str = config.data_dir

        self.data: pd.DataFrame = pd.read_csv(file_path)

        # get all of the unique timestamps with the data
        time_steps: pd.DataFrame = self.data.iloc[:, 0].unique()
        self.num_time_steps: int = time_steps.shape[0]

        self.point_clouds: list = []
        self.env_time_step: int = -1

        # get the point clouds for each time step
        for ts in range(self.num_time_steps):
            rows: pd.DataFrame = self.data[self.data.iloc[:, 0] == time_steps[ts]]
            pc: np.ndarray = rows.values[:, 1:4]
            self.point_clouds.append(pc)

    def step(self) -> np.ndarray:
        """
        Returns the next point cloud in the dataset.
        Raises a ValueError if the environment has reached the end of the data.

        Returns:
            np.ndarray: The next point cloud.
        """
        self.env_time_step += 1

        if self.env_time_step >= self.num_time_steps:
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
        return self.num_time_steps
    
    def reset(self) -> np.ndarray:
        """
        Resets the environment to its initial state and returns the first point cloud.

        Returns:
            np.ndarray: The first point cloud.
        """
        self.env_time_step = 0

        pc_dict = {
            "lidar_data": self.point_clouds[self.env_time_step][:, :2],
            "occupancy": self.point_clouds[self.env_time_step][:, 2]
        }

        return copy.deepcopy(pc_dict)        
