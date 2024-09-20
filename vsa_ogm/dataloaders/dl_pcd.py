from glob import glob
import numpy as np
from omegaconf import DictConfig
import PyntCloud
import os

def readPointCloud(file, intensity_threshold=None):
    """
    Read a point cloud from a file and apply an intensity threshold if specified.

    Args:
        file (str): The path to the point cloud file.
        intensity_threshold (float, optional): The intensity threshold to
            apply. Defaults to None.

    Returns:
        numpy.ndarray: A numpy array representing the point cloud, with one
            point per row and columns (x, y, z, i).
    """
    # numpy.ndarray with one point per row with columns (x, y, z, i)
    point_cloud = PyntCloud.from_file(file).points.values[:, 0:4]

    if intensity_threshold is not None:
        point_cloud[:, 3] = np.clip(
            a=point_cloud[:, 3] / intensity_threshold,
            a_min=0.0,
            a_max=1.0,
            dtype=np.float32
        )

    return point_cloud

class PcDataLoader:
    """
    A data loader for loading point cloud data.

    Args:
        config (DictConfig): Configuration dictionary containing data
        directory, file prefix, and file suffix.

    Attributes:
        files (list[str]): List of file paths matching the specified prefix
            and suffix.
        step_limit (int): Maximum index of the files list.
        time_step (int): Current index of the files list (-1 indicates uninitialized).
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the PCDDataLoader object.

        Args:
            config (DictConfig): Configuration dictionary containing data
            directory, file prefix, and file suffix.

        Attributes:
            files (list[str]): List of file paths matching the specified prefix
                and suffix.
            step_limit (int): Maximum index of the files list.
            time_step (int): Current index of the files list (-1 indicates uninitialized).
        """

        # -----------------------------
        # TODO Add Argument Validation
        # -----------------------------
        
        file_path: str = config.data_dir
        prefix: str = config.get("file_prefix", r"[0-9]")
        suffix: str = config.get("file_suffix", r".pcd")
        intensity_threshold: float = config.get("intensity_threshold", 100)

        mask: str = prefix + "*" + suffix
        file_path_mask: str = os.path.join(file_path, mask)
        files: list[str] = glob(file_path_mask)

        # sort the list of filepaths by the digits in their headers
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.files: list[str] = files
        self.step_limit: int = len(self.files) - 1
        self.time_step: int = -1

    def max_steps(self) -> int:
        """
        Returns the maximum number of steps allowed for the data loader.

        Returns:
            int: The maximum number of steps.
        """
        return self.step_limit

    def step(self, ts: int = None) -> np.ndarray:
        """
        Perform a step in the data loading process.

        Parameters:
            ts (int): The time step to load. If None, the current time step will be used.

        Returns:
            np.ndarray: The loaded point cloud data.

        Raises:
            AssertionError: If the time step is invalid or not an integer.
        """

        _ts: int = self.time_step

        if ts is not None:
            assert ts >= 0 and ts <= self.step_limit, "Invalid time step."
            assert isinstance(ts, int), "Time step must be an integer."
            _ts = ts

        fpath: str = self.files[_ts]
        point_cloud = readPointCloud(fpath, intensity_threshold=100)

        return point_cloud

    def reset(self) -> dict:
        """
        Resets the dataloader to its initial state.

        Returns:
            dict: The initial data sample at the initial time step.
        """

        self.time_step = 0

        return self.step()
