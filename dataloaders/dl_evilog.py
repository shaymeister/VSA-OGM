import cv2
from glob import glob
import numpy as np
from omegaconf import DictConfig
import os
from typing import Tuple

class EviLogDataLoader():
    """
    A data loader class for loading EviLog data.

    This class is responsible for loading input and labels data from the specified directories
    and creating file masks based on the provided configuration.

    Args:
        config (DictConfig): The configuration for the data loader.

    Attributes:
        data_dir (str): The directory where the data is stored.
        input_dir_name (str): The name of the directory where the input data is stored.
        labels_dir_name (str): The name of the directory where the labels data is stored.
        input_prefix (str): The file prefix for the input data.
        input_suffix (str): The file suffix for the input data.
        labels_prefix (str): The file prefix for the labels data.
        labels_suffix (str): The file suffix for the labels data.
        input_files (list[str]): The list of input file paths.
        labels_files (list[str]): The list of labels file paths.
    """
    def __init__(self, config: DictConfig) -> None:
        """
        Initializes a new instance of the EviLogDataLoader class.

        Args:
            config (DictConfig): The configuration for the data loader.
        """

        # -----------------------------
        # TODO Add Argument Validation
        # -----------------------------

        self.axis_resolution: float = config.get('axis_resolution', 0.1)

        data_dir: str = config.data_dir

        # Get the name of individual directories where the
        # inputs and labels are stored
        input_dir_name: str = config.get('input_dir_name', 'inputs')
        labels_dir_name: str = config.get('labels_dir_name', 'labels')

        # Get the file prefix and suffix for the input data
        # and combine them to create a mask for the input files
        input_prefix: str = config.get('input_prefix', r"[0-9]")
        input_suffix: str = config.get('input_suffix', r".pcd.png")
        input_mask: str = input_prefix + "*" + input_suffix

        # Get the file prefix and suffix for the labels data
        # and combine them to create a mask for the labels files
        labels_prefix: str = config.get('labels_prefix', r"[0-9]")
        labels_suffix: str = config.get('labels_suffix', r".pcd.png")
        labels_mask: str = labels_prefix + "*" + labels_suffix

        # Create the full paths to the input and labels directories
        input_dir_path: str = os.path.join(data_dir, input_dir_name)
        labels_dir_path: str = os.path.join(data_dir, labels_dir_name)

        # Create the full paths with masks to the input and labels files
        input_dir_file_mask: str = os.path.join(input_dir_path, input_mask)
        labels_dir_file_mask: str = os.path.join(labels_dir_path, labels_mask)

        assert os.path.exists(input_dir_path), f"Input directory does not exist: {input_dir_path}"
        assert os.path.exists(labels_dir_path), f"Labels directory does not exist: {labels_dir_path}"

        input_files: list[str] = glob(input_dir_file_mask)
        labels_files: list[str] = glob(labels_dir_file_mask)

        # sort the list of filepaths by the digits in their headers
        input_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        labels_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.input_files: list[str] = input_files
        self.labels_files: list[str] = labels_files
        self.time_step: int = -1

    def step(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a single step of the data loading process.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
                following:
                - all_points: An array of shape (N, 2) representing the
                    coordinates of all points.
                - labels: An array of shape (N,) representing the labels for
                    each point (1 for occupied, 0 for empty).
                - input_img: An array representing the input image.
                - labels_img: An array representing the labels image.
        """

        if self.time_step < 0:
            raise ValueError("Data loader not initialized.")
        
        if self.time_step >= self.max_steps():
            raise ValueError("No more samples to load.")
        
        input_path: str = self.input_files[self.time_step]
        labels_path: str = self.labels_files[self.time_step]

        input_img: np.ndarray = cv2.imread(input_path)
        labels_img: np.ndarray = cv2.imread(labels_path)

        input_img: np.ndarray = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        labels_img: np.ndarray = cv2.cvtColor(labels_img, cv2.COLOR_BGR2RGB)

        # The default color scheme for the lidar data points is [0, 0, 0] for
        # the background and [85, 0, 0] for the points. We will convert the
        # the labels to [0, 0, 0] for the background and [255, 0, 0] for each
        # individual lidar point.
        old_point_color: list = [0, 0, 0]
        new_point_color: list = [255, 0, 0]
        point_mask: np.ndarray = np.all(input_img == old_point_color, axis=2)
        input_img[~point_mask] = new_point_color

        # The shape of the label is smaller than the input image so we
        # need to resize the labels to match the input image.
        labels_img = cv2.resize(
            labels_img,
            (input_img.shape[1], input_img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # The default color for the empty points in [0, 85, 0] and the color
        # for the occupied points is [85, 0, 0]. We will convert the labels
        # to [0, 255, 0] and [255, 0, 0] respectively.
        old_red_color: list = [85, 0, 0]
        new_red_color: list = [255, 0, 0]
        old_green_color: list = [0, 85, 0]
        new_green_color: list = [0, 255, 0]
        red_mask: np.ndarray = np.all(labels_img == old_red_color, axis=2)
        green_mask: np.ndarray = np.all(labels_img == old_green_color, axis=2)
        labels_img[red_mask] = new_red_color
        labels_img[green_mask] = new_green_color

        # The default initial point cloud does not come with any labels.
        # We will create an approximate mask for the occupied and empty
        # from the label data.
        labeled_input = np.zeros_like(input_img)
        labeled_input = labeled_input.astype(np.uint8)

        # extract the red points from the label
        boolean_red_mask = np.all(labels_img == new_red_color, axis=2)
        labeled_input[boolean_red_mask] = new_red_color

        # blur the red points to avoid adding empty points over the occupied
        # points; we had to do this because the label points are not perfectly
        # aligned with the input points.
        input_blur = cv2.blur(labeled_input, (30, 30))
        info_mask = [0, 0, 0]
        boolean_info_mask = np.all(input_blur == info_mask, axis=2)

        # extract all points from the input image
        boolean_input_mask = np.all(input_img == new_point_color, axis=2)
        
        # assume that if a point is in the input image and in the info mask
        # then it is an empty point.
        empty_mask = np.logical_and(boolean_input_mask, boolean_info_mask)
        input_img[empty_mask] = [0, 255, 0]

        # We now need to extract the individual points from the input image
        # and create a binary mask for the occupied and empty points.
        boolean_red_mask = np.where(np.all(input_img == new_red_color, axis=2))
        boolean_green_mask = np.where(np.all(input_img == new_green_color, axis=2))

        occupied_points = np.concatenate([
            boolean_red_mask[1].reshape(-1, 1),
            boolean_red_mask[0].reshape(-1, 1)
        ], axis=1)

        free_points = np.concatenate([
            boolean_green_mask[1].reshape(-1, 1),
            boolean_green_mask[0].reshape(-1, 1)
        ], axis=1)

        occupied_points = occupied_points.astype(np.float32)
        free_points = free_points.astype(np.float32)

        occupied_points *= self.axis_resolution
        free_points *= self.axis_resolution

        all_points = np.concatenate([occupied_points, free_points], axis=0)
        labels = np.concatenate([
            np.ones(occupied_points.shape[0]),
            np.zeros(free_points.shape[0])
        ])

        self.time_step += 1

        return all_points, labels, input_img, labels_img
    
    def max_steps(self) -> int:
        """
        Returns the maximum number of steps that can be taken.

        Returns:
            int: The maximum number of steps that can be taken.
        """
        return len(self.input_files) - 1
    
    def reset(self) -> None:
        """
        Resets the data loader to the initial state.
        """
        self.time_step = 0
        return self.step()
