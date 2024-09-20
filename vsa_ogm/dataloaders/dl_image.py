import cv2
from glob import glob
import numpy as np
from omegaconf import DictConfig
import os

class ImageDataLoader:
    """
    A data loader class for loading image data.

    Args:
        config (DictConfig): Configuration dictionary containing data
        directory, file prefix, and file suffix.

    Attributes:
        files (list[str]): List of file paths.
        step_limit (int): Maximum index of the files list.
        time_step (int): Current index of the files list (-1 indicates
            uninitialized).
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the DLImage class.

        Args:
            config (DictConfig): Configuration dictionary containing data
            directory, file prefix, and file suffix.

        Attributes:
            files (list[str]): List of file paths.
            step_limit (int): Maximum index of the files list.
            time_step (int): Current index of the files list (-1 indicates
                uninitialized).
        """
       
        file_path: str = config.data_dir
        prefix: str = config.get("file_prefix", r"[0-9]")
        suffix: str = config.get("file_suffix", r".png")

        mask: str = prefix + "*" + suffix
        file_path_mask: str = os.path.join(file_path, mask)
        files: list[str] = glob(file_path_mask)

        # sort the list of filepaths by the digits in their headers
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.files: list[str] = files
        self.step_limit: int = len(self.files) - 1
        self.time_step: int = -1

    def step(self) -> np.ndarray:
        """
        Returns the next observation from the data loader.

        Returns:
            np.ndarray: The next observation.
        """
        if self.time_step > self.step_limit:
            raise ValueError("No more images to load.")
        
        fpath: str = self.files[self.time_step]
        image: np.ndarray = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.time_step += 1

        return image

    def max_steps(self) -> int:
        """
        Returns the maximum number of steps allowed for the data loader.

        Returns:
            int: The maximum number of steps.
        """
        return self.step_limit
    
    def reset(self) -> np.ndarray:
        """
        Resets the data loader to its initial state and returns the initial observation.
        """
        self.time_step = 0
        return self.step()