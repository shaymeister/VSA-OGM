from glob import glob
import numpy as np
from omegaconf import DictConfig
import os
from typing import Tuple

class ToySimDataLoader:
    """this class serves as a translation layer between ocg and data generated
    from the toy simulator

    the toy sim serves as a simple platform to test algorithms. the simulation
    consists of a cart with lidar traversing a set series of waypoints and
    mapping the surroundings as it does so.
    """

    def __init__(self, config: DictConfig):
        """
        Initializes a ToySimEnvData object.

        Args:
            config (DictConfig): A configuration object containing the following fields:
                - data_dir (str): The directory containing the data files.
                - file_prefix (str): The prefix of the data files.
                - file_suffix (str): The suffix of the data files.
                - map_size (list[int]): The size of the map.
        """

        # -----------------------------
        # TODO Add Argument Validation
        # -----------------------------

        file_path: str = config.data_dir
        prefix: str = config.file_prefix
        suffix: str = config.file_suffix

        mask: str = prefix + "*" + suffix
        file_path_mask: str = os.path.join(file_path, mask)
        files: list[str] = glob(file_path_mask)
        
        # sort the list of filepaths by the digits in their headers
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.files: list[str] = files
        self.map_size: list[int] = config.world_bounds
        self.step_limit: int = len(self.files) - 1
        self.time_step: int = -1

    def max_steps(self) -> int:
        """
        Returns the maximum number of steps allowed for the simulation.

        Returns:
            int: The maximum number of steps allowed for the simulation.
        """
        return self.step_limit

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resets the environment to its initial state and returns the initial
            observation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays
                representing the initial observation. The first array contains
                lidar data mapped onto a grid, and the second array is an
                empty map.
        """
        self.time_step = 0

        measurement_file = np.load(self.files[self.time_step])

        distance_data: np.ndarray = measurement_file["dist_theta_at_t"]
        laser_data: np.ndarray = measurement_file["laser_data_xy_at_t"]
        max_laser_distance: float = measurement_file["max_laser_distance"]
        pose_data: np.ndarray = measurement_file["all_robot_poses"][self.time_step, :]

        occupancy = np.zeros((laser_data.shape[0],))
        occupancy[np.where(distance_data != max_laser_distance)] = 1.0

        measurement_file.close()

        data_batch: dict = {
            "lidar_data": laser_data,
            "occupancy": occupancy,
            "lidar_distances": distance_data,
            "max_laser_distance": max_laser_distance,
            "robot_poses": pose_data
        }

        return data_batch

    def step(self) -> dict:
        """
        Loads the next measurement file and returns a dictionary containing
        the lidar data, lidar distances, occupancy, max laser distance, and
        robot poses.

        Returns:
            A dictionary containing the lidar data, lidar distances,
            occupancy, max laser distance, and robot poses.
        """
        self.time_step += 1

        measurement_file = np.load(self.files[self.time_step])

        distance_data: np.ndarray = measurement_file["dist_theta_at_t"]
        laser_data: np.ndarray = measurement_file["laser_data_xy_at_t"]
        max_laser_distance: float = measurement_file["max_laser_distance"]
        pose_data: np.ndarray = measurement_file["all_robot_poses"][self.time_step, :]

        measurement_file.close()

        occupancy = np.zeros((laser_data.shape[0],))
        occupancy[np.where(distance_data != max_laser_distance)] = 1.0

        data_batch: dict = {
            "lidar_data": laser_data,
            "lidar_distances": distance_data,
            "occupancy": occupancy,
            "max_laser_distance": max_laser_distance,
            "robot_poses": pose_data
        }

        return data_batch
