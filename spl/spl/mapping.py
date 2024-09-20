import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, ListConfig
import os
import time
import torch
from tqdm import tqdm
from typing import List, Tuple, Union

import spl.encoders as spe
import spl.functional as spf
from spl.generators import SSPGenerator
import spl.plotting as spp

def _validate_ogm2d_v4_init(config: DictConfig, world_size: Tuple[float],
                            log_dir: str) -> None:
    """
    Validate the initialization parameters for the OGM2D-V4 mapping.

    Args:
        config (DictConfig): The configuration dictionary.
        world_size (Tuple[float]): The world size tuple.
        log_dir (str): The log directory.

    Raises:
        AssertionError: If any of the validation checks fail.
    """
    assert isinstance(config, DictConfig)
    assert any(isinstance(config.axis_resolution, t) for t in [float, int])
    assert config.axis_resolution > 0
    assert any(isinstance(config.length_scale, t) for t in [float, int])
    assert config.length_scale > 0
    assert any(isinstance(config.quadrant_hierarchy, t) for t in [list, tuple, ListConfig])
    assert all(isinstance(level, int) for level in config.quadrant_hierarchy)
    assert all(level > 0 for level in config.quadrant_hierarchy)
    assert isinstance(config.use_query_normalization, bool)
    assert isinstance(config.use_query_rescaling, bool)
    assert isinstance(log_dir, str)
    assert os.path.exists(log_dir)
    assert any(isinstance(world_size, t) for t in [list, tuple, ListConfig])
    assert len(world_size) == 4
    assert all(any(isinstance(world_size[i], t) for t in [float, int]) for i in range(4))
    assert world_size[0] < world_size[1]
    assert world_size[2] < world_size[3]
    assert isinstance(config.vsa_dimensions, int)
    assert config.vsa_dimensions > 0
    assert isinstance(config.verbose, bool)
    assert isinstance(config.plotting.plot_xy_voxels, bool)


class OGM2D_V4:
    """
    This class represents a 2D occupancy grid mapper that uses spatial
    quadrants to encode the environment.
    """
    def __init__(self, config: DictConfig, world_size: Tuple[float],
            log_dir: str, plot_interval: int = 1) -> None:
        """
        Initializes a Mapping object with the given config and world size.

        Args:
            config (DictConfig): A configuration object containing the
                following fields:
                - TODO
            world_size (Tuple[float]): A tuple of four floats representing the
                bounds of the world in meters, in the following order:
                (xmin, xmax, ymin, ymax).
            log_dir (str): The directory to save the debugging files and plots
        """

        _validate_ogm2d_v4_init(config, world_size, log_dir)

        # ----------------
        # Class Constants
        # ----------------
        self.environment_dimensionality: int = 2
        
        # -----------------------------
        # Configuration Initialization
        # -----------------------------
        self.axis_resolution: float = config.axis_resolution
        self.decision_threshold: tuple = config.decision_thresholds
        self.device = torch.device(config.device)
        self.length_scale: float = config.length_scale
        self.log_dir: str = log_dir
        self.plotting_flags: dict = config.plotting
        self.quadrant_hierarchy: List[int] = config.quadrant_hierarchy
        self.use_query_normalization: bool = config.use_query_normalization
        self.use_query_rescaling: bool = config.use_query_rescaling
        self.verbose: bool = config.verbose
        self.vsa_dimensions: int = config.vsa_dimensions
        self.world_bounds: Tuple[float, float, float, float] = world_size

        self.plot_interval: int = config.get("plot_interval", 1)
        self.save_axis_matrix: bool = config.get("save_axis_matrix", False)
        self.use_matrix_encoding: bool = config.get("use_matrix_encoding", False)

        # --------------------------
        # Dependency Initialization
        # --------------------------
        self.pdist = torch.nn.PairwiseDistance()

        self.ssp_generator = SSPGenerator(
            dimensionality=self.vsa_dimensions,
            device=self.device,
            length_scale=self.length_scale
        )
        self.world_bounds_norm: Tuple[float, float] = (
            self.world_bounds[1] - self.world_bounds[0],
            self.world_bounds[3] - self.world_bounds[2]
        )

        # ----------------------------------
        # Variables for Observation Logging
        # ----------------------------------
        self.obs_count: int = 0
        self.obs_hist_len: int = 1000
        self.obs_log: dict = {
            "runtime": {
                "encoding": np.zeros(shape=(self.obs_hist_len,)),
                "occupancy_probability_calculation": np.zeros(shape=(self.obs_hist_len,)),
                "empty_probability_calculation": np.zeros(shape=(self.obs_hist_len,)),
                "pc_normalization": np.zeros(shape=(self.obs_hist_len,)),
                "quadrant_center_calculation": np.zeros(shape=(self.obs_hist_len,)),
            },
            "point_clouds": {
                "originals": [],
                "originals_normalized": [],
            },
            "obs_proc": {
                "quadrant_centers_matrix": [],
                "quadrant_memory_vectors": [],
                "occupied_heatmaps": [],
                "occupied_query_vectors": [],
                "empty_heatmaps": [],
                "empty_query_vectors": [],
            }
        }

        # ----------------------------------
        # empty variables for class methods
        # ----------------------------------
        self.quadrant_axis_bounds: Tuple[Tuple[torch.tensor, torch.tensor]] = []
        self.quadrant_centers: Tuple[torch.tensor] = []
        self.occupied_quadrant_memory_vectors: torch.tensor = None
        self.empty_quadrant_memory_vectors: torch.tensor = None
        self.xy_axis_linspace: tuple[torch.tensor] = []
        self.xy_axis_vectors: torch.tensor = None
        self.xy_axis_matrix: torch.tensor = None
        self.xy_axis_global_heatmap: torch.tensor = None
        self.xy_axis_occupied_heatmap: torch.tensor = None
        self.xy_axis_empty_heatmap: torch.tensor = None
        self.xy_axis_class_matrix: torch.tensor = None

        self._build_quadrant_hierarchy()
        self._build_quadrant_memory_hierarchy()
        self._build_quadrant_indices()

        if len(self.quadrant_hierarchy) != len(self.quadrant_axis_bounds):
            raise ValueError("The number of quadrant hierarchy levels must match the number of quadrant axis bounds tuples.")
        
        if len(self.quadrant_hierarchy) != len(self.quadrant_centers):
            raise ValueError("The number of quadrant hierarchy levels must match the number of quadrant center grids.")
        
        self._build_xy_axis_linspace()
        self._build_xy_axis_vectors()
        self._build_xy_axis_matrix()
        self._build_xy_axis_heatmaps()
        self._build_xy_axis_class_matrices()
    
    def process_observation(self, point_cloud: Union[np.ndarray, torch.tensor],
            labels: Union[np.ndarray, torch.tensor]) -> None:
        """
        Processes an observation represented as a point cloud and the
        corresponding labels for each point.

        Args:
            point_cloud (Union[np.ndarray, torch.tensor]): A 2D tensor of points
            labels (Union[np.ndarray, torch.tensor]): A 1D tensor of labels

        Returns:
            None
        """
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud)
        
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        
        if point_cloud.device != self.device:
            point_cloud = point_cloud.to(self.device)
        
        if labels.device != self.device:
            labels = labels.to(self.device)

        # Log the original and unmodified point cloud
        ogpc: torch.tensor = point_cloud.clone()
        ogpc: np.ndarray = ogpc.detach().cpu().numpy()
        self.obs_log["point_clouds"]["originals"].append(ogpc)

        # ----------------------------------------------
        # normalize the point cloud to the world bounds
        # ----------------------------------------------
        norm_time_start: float = time.time()
        
        point_cloud[:, 0] -= self.world_bounds[0]
        point_cloud[:, 1] -= self.world_bounds[2]

        norm_time_end: float = time.time()
        norm_time: float = norm_time_end - norm_time_start
        self.obs_log["runtime"]["pc_normalization"][self.obs_count] += norm_time

        # log the original point cloud normalized to the world bounds
        ogpcn: torch.tensor = point_cloud.clone()
        ogpcn: np.ndarray = ogpcn.detach().cpu().numpy()
        self.obs_log["point_clouds"]["originals_normalized"].append(ogpcn)
    
        ups = point_cloud
        ups_labels = labels
        
        # -----------------------------------------------
        # Calculate quadrant memories for each new point
        # using a multipoint L2 distance calculation
        # -----------------------------------------------
        qcc_time_start: float = time.time()

        ups: torch.tensor = ups.unsqueeze(1)
        qcm: torch.tensor = self.quadrant_centers[0]
        qcm: torch.tensor = qcm.unsqueeze(0)
        qcm: torch.tensor = qcm.repeat(ups.shape[0], 1, 1)
        dists: torch.tensor = self.pdist(ups, qcm)
        closest_quads: torch.tensor = torch.argmin(dists, dim=1)

        qcc_time_end: float = time.time()
        qcc_time: float = qcc_time_end - qcc_time_start
        self.obs_log["runtime"]["quadrant_center_calculation"][self.obs_count] += qcc_time

        assert len(ups.shape) == 3
        assert len(qcm.shape) == 3
        assert ups.shape[0] == qcm.shape[0]
        assert ups.shape[0] == ups_labels.shape[0]
        assert ups.shape[1] == 1
        assert ups.shape[2] == self.environment_dimensionality
        assert qcm.shape[1] == self.quadrant_hierarchy[0] ** self.environment_dimensionality
        assert qcm.shape[2] == self.environment_dimensionality

        # log the quadrant centers matrix
        qcm_copy: torch.tensor = qcm.clone()
        qcm_copy: np.ndarray = qcm_copy.detach().cpu().numpy()
        self.obs_log["obs_proc"]["quadrant_centers_matrix"].append(qcm_copy)

        ups = ups.squeeze(1)
    
        # ---------------------------------------------------
        # Convert the new points to hyperdimensional vectors
        # ---------------------------------------------------
        encoding_time_start: float = time.time()

        if self.use_matrix_encoding:
            # ---------------------------
            # Matrix Encoding Approach
            # ---------------------------
            x_axis_fd = torch.fft.fft(self.xy_axis_vectors[0])
            y_axis_fd = torch.fft.fft(self.xy_axis_vectors[1])

            x_axis_fd_matrix = x_axis_fd.unsqueeze(0).repeat(ups.shape[0], 1)
            y_axis_fd_matrix = y_axis_fd.unsqueeze(0).repeat(ups.shape[0], 1)

            x_powers = (ups[:, 0] / self.length_scale)
            y_powers = (ups[:, 1] / self.length_scale)

            x_power_matrix = x_powers.repeat(self.vsa_dimensions, 1).T
            y_power_matrix = y_powers.repeat(self.vsa_dimensions, 1).T

            x_axis_fd_matrix = x_axis_fd_matrix ** x_power_matrix
            y_axis_fd_matrix = y_axis_fd_matrix ** y_power_matrix

            x_axis_fd_matrix = x_axis_fd_matrix.unsqueeze(0)
            y_axis_fd_matrix = y_axis_fd_matrix.unsqueeze(0)
            
            xy_axis_fd_matrix = torch.concatenate((x_axis_fd_matrix, y_axis_fd_matrix), dim=0)

            xy_axis_fd_matrix = torch.prod(xy_axis_fd_matrix, dim=0)

            xy_axis_fd_matrix = torch.fft.ifft(xy_axis_fd_matrix, dim=1)
            xy_axis_fd_matrix = xy_axis_fd_matrix.real

            unique_quads = torch.unique(closest_quads)
            for quad in unique_quads:
                occ_point_mask = torch.where(((closest_quads == quad) & (ups_labels == 1)))
                emp_point_mask = torch.where(((closest_quads == quad) & (ups_labels == 0)))

                occ_point_vectors = xy_axis_fd_matrix[occ_point_mask]
                emp_point_vectors = xy_axis_fd_matrix[emp_point_mask]
            
                if occ_point_vectors.shape[0] > 0:
                    occupied_point_bundle = torch.sum(occ_point_vectors, dim=0)
                    self.occupied_quadrant_memory_vectors[quad] += occupied_point_bundle

                if emp_point_vectors.shape[0] > 0:
                    empty_point_bundle = torch.sum(emp_point_vectors, dim=0)
                    self.empty_quadrant_memory_vectors[quad] += empty_point_bundle
        else:
            # ---------------------------
            # For-Loop Encoding Approach
            # ---------------------------
            occupied_points = ups[ups_labels == 1]
            occupied_points_closest_quads = closest_quads[ups_labels == 1]

            for idx, point in enumerate(occupied_points):
                vs: list[torch.tensor] = [
                    spf.power(self.xy_axis_vectors[0], point[0], self.length_scale),
                    spf.power(self.xy_axis_vectors[1], point[1], self.length_scale),
                ]
                point_vector: torch.tensor = spf.bind(vs, self.device)
                self.occupied_quadrant_memory_vectors[
                    occupied_points_closest_quads[idx],
                ] += point_vector   
            
            empty_points = ups[ups_labels == 0]
            empty_points_closest_quads = closest_quads[ups_labels == 0]
            for idx, point in enumerate(empty_points):
                vs: list[torch.tensor] = [
                    spf.power(self.xy_axis_vectors[0], point[0], self.length_scale),
                    spf.power(self.xy_axis_vectors[1], point[1], self.length_scale),
                ]
                point_vector: torch.tensor = spf.bind(vs, self.device)
                self.empty_quadrant_memory_vectors[
                    empty_points_closest_quads[idx]
                ] += point_vector


        encoding_time_end: float = time.time()
        encoding_time: float = encoding_time_end - encoding_time_start
        self.obs_log["runtime"]["encoding"][self.obs_count] += encoding_time

        # log the quadrant memory vectors
        qmv_copy: torch.tensor = self.empty_quadrant_memory_vectors.clone()
        qmv_copy: np.ndarray = qmv_copy.detach().cpu().numpy()
        self.obs_log["obs_proc"]["quadrant_memory_vectors"].append(qmv_copy)

        # ----------------------------------------------
        # Calculate the probabilities of occupancy for
        # each point in the xy axis matrix
        # ----------------------------------------------
        occ_prob_calc_time_start: float = time.time()
                
        counter = 0
        for j, y_lower in enumerate(self.quadrant_indices_y[:-1]):
            for i, x_lower in enumerate(self.quadrant_indices_x[:-1]):
                x_upper = self.quadrant_indices_x[i + 1]
                y_upper = self.quadrant_indices_y[j + 1]

                xy_axis_matrix_quad = self.xy_axis_matrix[x_lower:x_upper, y_lower:y_upper, :]

                qv = self.occupied_quadrant_memory_vectors[counter, :]
                qv = qv / torch.linalg.norm(qv)
                    
                quadrant_heatmap = torch.tensordot(
                    qv,
                    xy_axis_matrix_quad,
                    dims=([0], [2])
                )

                quadrant_heatmap /= torch.max(quadrant_heatmap)

                self.xy_axis_occupied_heatmap[x_lower:x_upper, y_lower:y_upper] = quadrant_heatmap
                
                counter += 1

        occ_prob_calc_time_end: float = time.time()
        occ_prob_calc_time: float = occ_prob_calc_time_end - occ_prob_calc_time_start
        self.obs_log["runtime"]["occupancy_probability_calculation"][self.obs_count] += occ_prob_calc_time

        self.xy_axis_occupied_heatmap = self.xy_axis_occupied_heatmap.T
        self.xy_axis_occupied_heatmap = torch.nan_to_num(self.xy_axis_occupied_heatmap)
        self.xy_axis_occupied_heatmap = self.xy_axis_occupied_heatmap / torch.max(self.xy_axis_occupied_heatmap)
        
        # log the occupied heatmap
        ohm_copy: torch.tensor = self.xy_axis_occupied_heatmap.clone()
        ohm_copy: np.ndarray = ohm_copy.detach().cpu().numpy()
        self.obs_log["obs_proc"]["occupied_heatmaps"].append(ohm_copy)

        # log the occupied query vectors
        oqv_copy: torch.tensor = self.occupied_quadrant_memory_vectors.clone()
        oqv_copy: np.ndarray = oqv_copy.detach().cpu().numpy()
        self.obs_log["obs_proc"]["occupied_query_vectors"].append(oqv_copy)

        # ----------------------------------------------
        # Calculate the probabilities of empty for
        # each point in the xy axis matrix
        # ----------------------------------------------
        empty_prob_calc_time_start: float = time.time()

        counter = 0
        for j, y_lower in enumerate(self.quadrant_indices_y[:-1]):
            for i, x_lower in enumerate(self.quadrant_indices_x[:-1]):
                x_upper = self.quadrant_indices_x[i + 1]
                y_upper = self.quadrant_indices_y[j + 1]

                xy_axis_matrix_quad = self.xy_axis_matrix[x_lower:x_upper, y_lower:y_upper, :]

                qv = self.empty_quadrant_memory_vectors[counter, :]
                qv = qv / torch.linalg.norm(qv)

                quadrant_heatmap = torch.tensordot(
                    qv,
                    xy_axis_matrix_quad,
                    dims=([0], [2])
                )

                quadrant_heatmap /= torch.max(quadrant_heatmap)

                self.xy_axis_empty_heatmap[x_lower:x_upper, y_lower:y_upper] = quadrant_heatmap
                
                counter += 1

        empty_prob_calc_time_end: float = time.time()
        empty_prob_calc_time: float = empty_prob_calc_time_end - empty_prob_calc_time_start
        self.obs_log["runtime"]["empty_probability_calculation"][self.obs_count] += empty_prob_calc_time

        self.xy_axis_empty_heatmap = self.xy_axis_empty_heatmap.T
        self.xy_axis_empty_heatmap = torch.nan_to_num(self.xy_axis_empty_heatmap)
        self.xy_axis_empty_heatmap = self.xy_axis_empty_heatmap / torch.max(self.xy_axis_empty_heatmap)

        # log the empty heatmap
        ehm_copy: torch.tensor = self.xy_axis_empty_heatmap.clone()
        ehm_copy: np.ndarray = ehm_copy.detach().cpu().numpy()
        self.obs_log["obs_proc"]["empty_heatmaps"].append(ehm_copy)

        # log the empty query vectors
        eqv_copy: torch.tensor = self.empty_quadrant_memory_vectors.clone()
        eqv_copy: np.ndarray = eqv_copy.detach().cpu().numpy()
        self.obs_log["obs_proc"]["empty_query_vectors"].append(eqv_copy)

        if self.obs_count % self.plot_interval == 0:
            self._plot_observation()

        self.obs_count += 1


    def get_global_heatmap(self) -> torch.tensor:
        """
        Returns the current global heatmap.

        Args:
            None
        
        Returns:
            torch.tensor: The current heatmap.
        """
        return self.xy_axis_global_heatmap
    
    def get_occupied_heatmap(self) -> torch.tensor:
        """
        Returns the current occupied heatmap.

        Args:
            None
        
        Returns:
            torch.tensor: The current heatmap.
        """
        return self.xy_axis_occupied_heatmap
    
    def get_empty_heatmap(self) -> torch.tensor:
        """
        Returns the current empty heatmap.

        Args:
            None
        
        Returns:
            torch.tensor: The current heatmap.
        """
        return self.xy_axis_empty_heatmap
            
    def get_class_matrix(self) -> torch.tensor:
        """
        Returns the current class matrix.

        Args:
            None
        
        Returns:
            torch.tensor: The current class matrix.
        """
        return self.xy_axis_class_matrix

    def query_point_thetas(self, points: Union[np.ndarray, torch.tensor],
                return_as_numpy: bool = True) -> torch.tensor:
        """
        Queries the memory for the given point and returns the theta value.

        Args:
            - points: A numpy array or torch tensor representing the points to
                query.
            - return_as_numpy: A boolean indicating whether to return the
                results as a numpy array (default: True).

        Returns:
            - results: A torch tensor or numpy array containing the theta
                values for the queried points.
        """
        assert isinstance(points, np.ndarray) or isinstance(points, torch.Tensor)
        assert len(points.shape) == 2
        assert points.shape[1] == self.environment_dimensionality
        assert isinstance(return_as_numpy, bool)

        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        
        if points.device != self.device:
            points = points.to(self.device)

        assert torch.min(points[:, 0]) >= self.world_bounds[0]
        assert torch.max(points[:, 0]) <= self.world_bounds[1]
        assert torch.min(points[:, 1]) >= self.world_bounds[2]
        assert torch.max(points[:, 1]) <= self.world_bounds[3]

        points[:, 0] -= self.world_bounds[0]
        points[:, 1] -= self.world_bounds[2]
        points = points / self.axis_resolution
        points = torch.round(points)
        points = points.long()
        
        results: torch.tensor = self.xy_axis_global_heatmap[points[:, 0], points[:, 1]]

        assert len(results.shape) == 1
        assert results.shape[0] == points.shape[0]

        if return_as_numpy:
            results = results.detach().cpu().numpy()
        
        return results
            
    def query_point_classes(self, points: Union[np.ndarray, torch.tensor],
                return_as_numpy: bool = True) -> torch.tensor:
        """
        Queries the memory for the given point and returns the class.

        Args:
            point (torch.tensor): The point to query.

        Returns:
            torch.tensor: The class for the given point.
        """
        assert isinstance(points, np.ndarray) or isinstance(points, torch.Tensor)
        assert len(points.shape) == 2
        assert points.shape[1] == self.environment_dimensionality
        assert isinstance(return_as_numpy, bool)

        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points)
        
        if points.device != self.device:
            points = points.to(self.device)

        assert torch.min(points[:, 0]) >= self.world_bounds[0]
        assert torch.max(points[:, 0]) <= self.world_bounds[1]
        assert torch.min(points[:, 1]) >= self.world_bounds[2]
        assert torch.max(points[:, 1]) <= self.world_bounds[3]

        points[:, 0] -= self.world_bounds[0]
        points[:, 1] -= self.world_bounds[2]
        points = points / self.axis_resolution
        points = torch.round(points)
        points = points.long()
        
        results: torch.tensor = self.xy_axis_class_matrix[points[:, 0], points[:, 1]]

        assert len(results.shape) == 1
        assert results.shape[0] == points.shape[0]

        if return_as_numpy:
            results = results.detach().cpu().numpy()
        
        return results
    
    def _plot_observation(self) -> None:
        """
        Plot the observation data and save the plots in the log directory.

        This method creates a directory for the current observation in the log
        directory, and then calls several other methods to plot and save
        various aspects of the observation data.
        """

        base_path = os.path.join(
            self.log_dir,
            "observations",
            f"observation_{self.obs_count}"
        )
        os.makedirs(base_path, exist_ok=True)

        self._plot_runtimes(base_path)
        self._plot_total_runtimes(base_path)
        self._plot_point_counts(base_path)
        self._plot_point_clouds(base_path)
        self._plot_obs_proc(base_path)

    def _plot_obs_proc(self, base_path: str, obs_proc_dir: str = "obs_proc",
            plots_dir: str = "plots", numpy_dir: str = "numpy_arrays") -> None:
        """
        Plot and save observation processing data.

        Args:
            base_path (str): The base path where the observation processing
                data will be saved.
            obs_proc_dir (str, optional): The directory name for the
                observation processing data. Defaults to "obs_proc".
            plots_dir (str, optional): The directory name for the plots.
                Defaults to "plots".
            numpy_dir (str, optional): The directory name for the numpy arrays.
                Defaults to "numpy_arrays".
        """
        assert isinstance(base_path, str)
        assert isinstance(obs_proc_dir, str)
        assert isinstance(plots_dir, str)
        assert isinstance(numpy_dir, str)
        assert os.path.exists(base_path)

        obs_proc_path: str = os.path.join(
            base_path,
            obs_proc_dir
        )
        os.makedirs(obs_proc_path, exist_ok=True)
        assert os.path.exists(obs_proc_path)

        plots_path: str = os.path.join(
            obs_proc_path,
            plots_dir,
        )
        os.makedirs(plots_path, exist_ok=True)
        assert os.path.exists(plots_path)

        numpy_path: str = os.path.join(
            obs_proc_path,
            numpy_dir,
        )
        os.makedirs(numpy_path, exist_ok=True)
        assert os.path.exists(numpy_path)
            
        data_headers: list[str] = [
            "quadrant_centers_matrix",
            "quadrant_memory_vectors",
            "occupied_heatmaps",
            "empty_heatmaps",
        ]
        
        for header in data_headers:
            header_path: str = os.path.join(
                numpy_path,
                header + ".npy"
            )
        
            with open(header_path, "wb") as f:
                np.save(f, self.obs_log["obs_proc"][header][self.obs_count])
            
        plot_headers: list[str] = [
            "occupied_heatmaps",
            "empty_heatmaps",
        ]
        
        plot_titles: list[str] = [
            "Occupied Heatmap",
            "Empty Heatmap",
        ]

        assert len(plot_headers) == len(plot_titles)
            
        for i in range(len(plot_headers)):
            header = plot_headers[i]
            title = plot_titles[i]

            header_sp_png: str = os.path.join(
                plots_path,
                header + ".png"
            )
            spp.plot_2d_heatmap_queried(
                plane_matrix=self.obs_log["obs_proc"][header][self.obs_count],
                axis_bounds=[
                    0, self.world_bounds_norm[0],
                    0, self.world_bounds_norm[1]
                ],
                save_path=header_sp_png,
                title=title,
            )

            header_sp_npy: str = os.path.join(
                numpy_path,
                header + ".npy"
            )
            with open(header_sp_npy, "wb") as f:
                np.save(f, self.obs_log["obs_proc"][header][self.obs_count])
    
    def _plot_point_counts(self, base_path: str, point_counts_dir: str = "point_counts",
            plots_dir: str = "plots", numpy_dir: str = "numpy_arrays") -> None:
        """
        Plot the point counts for different point cloud types and save the
        plots and numpy arrays.

        Args:
            base_path (str): The base path where the point counts, plots, and
                numpy arrays will be saved.
            point_counts_dir (str, optional): The directory name for the point
                counts. Defaults to "point_counts".
            plots_dir (str, optional): The directory name for the plots.
                Defaults to "plots".
            numpy_dir (str, optional): The directory name for the numpy arrays.
                Defaults to "numpy_arrays".
        """
        assert isinstance(base_path, str)
        assert isinstance(point_counts_dir, str)
        assert isinstance(plots_dir, str)
        assert isinstance(numpy_dir, str)
        assert os.path.exists(base_path)

        counts_path: str = os.path.join(
            base_path,
            point_counts_dir
        )        
        os.makedirs(counts_path, exist_ok=True)
        assert os.path.exists(counts_path)

        plots_path: str = os.path.join(
            counts_path,
            plots_dir,
        )        
        os.makedirs(plots_path, exist_ok=True)   
        assert os.path.exists(plots_path)

        numpy_path: str = os.path.join(
            counts_path,
            numpy_dir,
        )
        os.makedirs(numpy_path, exist_ok=True)
        assert os.path.exists(numpy_path)

        path_headers: list[str] = [
            "originals",
            "originals_normalized",
        ]

        plot_titles: list[str] = [
            "Original",
            "Original Normalized",
        ]

        assert len(path_headers) == len(plot_titles)

        for i in range(len(path_headers)):
            header = path_headers[i]
            title = plot_titles[i]

            counts = np.zeros(shape=(self.obs_hist_len,))
            for j in range(self.obs_hist_len)[:self.obs_count + 1]:
                
                # skip the current observation if it doesn't exist
                if j >= len(self.obs_log["point_clouds"][header]):
                    continue
                
                counts[j] = self.obs_log["point_clouds"][header][j].shape[0]

            header_sp_png: str = os.path.join(
                plots_path,
                header + ".png"
            )
            spp.plot_1d_numpy(
                x=counts,
                save_path=header_sp_png,
                title=title,
                y_label="Point Count",
            )

            header_sp_npy: str = os.path.join(
                numpy_path,
                header + ".npy"
            )
            with open(header_sp_npy, "wb") as f:
                np.save(f, counts)
    
    def _plot_total_runtimes(self, base_path: str, runtimes_dir: str = "runtimes",
            global_ph: str = "total_runtimes", plots_dir: str = "plots",
            numpy_dir: str = "numpy_arrays") -> None:
        """
        Plot the total runtimes and save the plots and numpy arrays.

        Args:
            base_path (str): The base path where the runtimes directory will
                be created.
            runtimes_dir (str, optional): The name of the runtimes directory.
                Defaults to "runtimes".
            global_ph (str, optional): The name of the global plot and numpy
                array. Defaults to "total_runtimes".
            plots_dir (str, optional): The name of the plots directory inside
                the runtimes directory. Defaults to "plots".
            numpy_dir (str, optional): The name of the numpy arrays directory
                inside the runtimes directory. Defaults to "numpy_arrays".

        Returns:
            None
        """

        assert isinstance(base_path, str)
        assert isinstance(runtimes_dir, str)
        assert os.path.exists(base_path)
        assert isinstance(global_ph, str)

        runtime_path: str = os.path.join(
            base_path,
            runtimes_dir
        )
        os.makedirs(runtime_path, exist_ok=True)
        assert os.path.exists(runtime_path)

        # accumulate the total runtimes
        total_times = np.zeros(shape=(self.obs_hist_len,))
        for key in self.obs_log["runtime"].keys():
            total_times += self.obs_log["runtime"][key]

        global_path_plots: str = os.path.join(
            runtime_path,
            plots_dir
        )
        os.makedirs(global_path_plots, exist_ok=True)
        assert os.path.exists(global_path_plots)

        global_path_numpy: str = os.path.join(
            runtime_path,
            numpy_dir
        )
        os.makedirs(global_path_numpy, exist_ok=True)
        assert os.path.exists(global_path_numpy)

        global_ph_png: str = os.path.join(
            global_path_plots,
            global_ph + ".png"
        )
        spp.plot_1d_numpy(
            x=total_times,
            save_path=global_ph_png,
            title="Total Runtimes",
            y_label="Time (s)",
        )

        global_ph_npy: str = os.path.join(
            global_path_numpy,
            global_ph + ".npy"
        )
        with open(global_ph_npy, "wb") as f:
            np.save(f, total_times)
    
    def _plot_runtimes(self, base_path: str, runtimes_dir: str = "runtimes",
            plots_dir: str = "plots", numpy_dir: str = "numpy_arrays") -> None:
        """
        Plot runtimes for different stages of the mapping process and save
        the plots and numpy arrays.

        Args:
            base_path (str): The base path for each directory.
            runtimes_dir (str, optional): The name of the directory to
                store the runtimes. Defaults to "runtimes".
            plots_dir (str, optional): The name of the directory to store
                the plots. Defaults to "plots".
            numpy_dir (str, optional): The name of the directory to store
                the numpy arrays. Defaults to "numpy_arrays".
        """
        assert isinstance(base_path, str)
        assert isinstance(runtimes_dir, str)
        assert isinstance(plots_dir, str)
        assert isinstance(numpy_dir, str)
        assert os.path.exists(base_path)

        empty_qmv = torch.clone(self.empty_quadrant_memory_vectors)
        occupied_qmv = torch.clone(self.occupied_quadrant_memory_vectors)
        
        occupied_qmv_sp = os.path.join(
            base_path,
            "occupied_quadrant_memory_vectors.npy"
        )
        with open(occupied_qmv_sp, "wb") as f:
            np.save(f, occupied_qmv.detach().cpu().numpy())

        empty_qmv_sp = os.path.join(
            base_path,
            "empty_quadrant_memory_vectors.npy"
        )
        with open(empty_qmv_sp, "wb") as f:
            np.save(f, empty_qmv.detach().cpu().numpy())
    
        runtime_path: str = os.path.join(
            base_path,
            runtimes_dir
        )        
        os.makedirs(runtime_path, exist_ok=True)
        assert os.path.exists(runtime_path)
    
        plots_path: str = os.path.join(
            runtime_path,
            plots_dir,
        )        
        os.makedirs(plots_path, exist_ok=True)   
        assert os.path.exists(plots_path)
            
        numpy_path: str = os.path.join(
            runtime_path,
            numpy_dir,
        )
        os.makedirs(numpy_path, exist_ok=True)
        assert os.path.exists(numpy_path)
            
        path_headers: list[str] = [
            "encoding",
            "occupancy_probability_calculation",
            "empty_probability_calculation",
            "pc_normalization",
            "quadrant_center_calculation",
        ]
        plot_titles: list[str] = [
            "Encoding",
            "Occupancy Probability Calculation",
            "Empty Probability Calculation",
            "Point Cloud Normalization",
            "Quadrant Center Calculation",
        ]

        assert len(path_headers) == len(plot_titles)
        assert len(path_headers) == len(self.obs_log["runtime"].keys())

        for i in range(len(path_headers)):
            header = path_headers[i]
            title = plot_titles[i]
        
            header_sp_png: str = os.path.join(
                plots_path,
                header + ".png"
            )
            spp.plot_1d_numpy(
                x=self.obs_log["runtime"][header],
                save_path=header_sp_png,
                title=title,
            )

            header_sp_npy: str = os.path.join(
                numpy_path,
                header + ".npy"
            )
            with open(header_sp_npy, "wb") as f:
                np.save(f, self.obs_log["runtime"][header])
    
    def _plot_point_cloud(self, point_cloud: np.ndarray,
            clear_figure: bool = True, save_path: str = None,
            show_plot: bool = False, normalized: bool = True,
            title: str = "Point Cloud", x_label: str = "X",
            y_label: str = "Y", colors: np.ndarray = None) -> None:
        """
        Plot a point cloud.

        Args:
            point_cloud (np.ndarray): The point cloud to plot.
            clear_figure (bool, optional): Whether to clear the figure before
                plotting. Defaults to True.
            save_path (str, optional): The file path to save the plot. Defaults
                to None.
            show_plot (bool, optional): Whether to display the plot. Defaults
                to False.
            normalized (bool, optional): Whether to normalize the plot.
                Defaults to True.
            title (str, optional): The title of the plot. Defaults to
                "Point Cloud".
            x_label (str, optional): The label for the x-axis. Defaults to "X".
            y_label (str, optional): The label for the y-axis. Defaults to "Y".
            colors (np.ndarray, optional): The colors for each point in the
                point cloud. Defaults to None.

        Raises:
            AssertionError: If the input arguments are not of the expected
                types or shapes.

        Returns:
            None
        """
        assert isinstance(point_cloud, np.ndarray)
        assert len(point_cloud.shape) == 2
        assert point_cloud.shape[1] == self.environment_dimensionality
        assert isinstance(clear_figure, bool)
        assert isinstance(save_path, str) or save_path is None
        assert isinstance(show_plot, bool)
        assert isinstance(normalized, bool)
        assert isinstance(title, str)
        assert isinstance(x_label, str)
        assert isinstance(y_label, str)
        assert isinstance(colors, np.ndarray) or colors is None

        if clear_figure:
            plt.clf()

        if colors is not None:
            assert colors.shape[0] == point_cloud.shape[0]
            assert len(colors.shape) == 1
            plt.scatter(
                point_cloud[:, 0],
                point_cloud[:, 1],
                c=colors
            )
            plt.colorbar()
        else:
            plt.scatter(
                point_cloud[:, 0],
                point_cloud[:, 1]
            )

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if normalized:
            plt.xlim(0, self.world_bounds_norm[0])
            plt.ylim(0, self.world_bounds_norm[1])
        else:
            plt.xlim(self.world_bounds[0], self.world_bounds[1])
            plt.ylim(self.world_bounds[2], self.world_bounds[3])

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        if show_plot:
            plt.show()
    
    def _plot_point_clouds(self, base_path: str, point_clouds_dir: str = "point_clouds",
            plots_dir: str = "plots", numpy_dir: str = "numpy_arrays") -> None:
        """
        TODO Finish Documentation
        """
        assert isinstance(base_path, str)
        assert isinstance(point_clouds_dir, str)
        assert os.path.exists(base_path)
        
        point_clouds_path: str = os.path.join(
            base_path,
            point_clouds_dir
        )

        os.makedirs(point_clouds_path, exist_ok=True)
        assert os.path.exists(point_clouds_path)

        plots_path: str = os.path.join(
            point_clouds_path,
            plots_dir,
        )
        os.makedirs(plots_path, exist_ok=True)
        assert os.path.exists(plots_path)
            
        numpy_path: str = os.path.join(
            point_clouds_path,
            numpy_dir,
        )
        os.makedirs(numpy_path, exist_ok=True)
        assert os.path.exists(numpy_path)
        
        # Plot the original point cloud
        original_plot_path: str = os.path.join(
            plots_path,
            "original.png"
        )
        self._plot_point_cloud(
            point_cloud=self.obs_log["point_clouds"]["originals"][self.obs_count],
            save_path=original_plot_path,
            title="Original Point Cloud",
            normalized=False
        )
        original_numpy_path: str = os.path.join(
            numpy_path,
            "original.npy"
        )
        with open(original_numpy_path, "wb") as f:
            np.save(f, self.obs_log["point_clouds"]["originals"][self.obs_count])

        # Plot the original point clouds before calculating unique points
        key_list: list[str] = [
            "originals_normalized",
        ]

        title_list: list[str] = [
            "Original Normalized",
        ]

        assert len(key_list) == len(title_list)

        for idx, key in enumerate(key_list):
            assert key in self.obs_log["point_clouds"].keys()

            # skip the current observation log if it doesn't exist
            if self.obs_count >= len(self.obs_log["point_clouds"][key]):
                continue
    
            plot_path = os.path.join(
                plots_path,
                f"{key}.png"
            )
            self._plot_point_cloud(
                point_cloud=self.obs_log["point_clouds"][key][self.obs_count],
                save_path=plot_path,
                title=key_list[idx]
            )

            np_path = os.path.join(
                numpy_path, 
                f"{key}.npy"
            )
            with open(np_path, "wb") as f:
                np.save(f, self.obs_log["point_clouds"][key][self.obs_count])

    def _build_xy_axis_class_matrices(self) -> None:
        """
        Builds the XY axis class matrices.

        This method asserts that the `xy_axis_matrix` attribute is not None, is
        of type `torch.tensor`, and has a shape with three dimensions. It then
        initializes the `xy_axis_heatmap` attribute as a tensor of zeros with
        the same shape as `xy_axis_matrix`.

        Args:
            None
            
        Returns:
            None
        
        Raises:
            AssertionError: If xy_axis_matrix is None, not a torch.tensor,
            or has an invalid shape.
        """
        assert self.xy_axis_matrix is not None
        assert isinstance(self.xy_axis_matrix, torch.Tensor)
        assert len(self.xy_axis_matrix.shape) == 3

        if self.verbose:
            print("Building XY axis class matrices...")

        self.xy_axis_class_matrix = torch.ones(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )
        self.xy_axis_class_matrix *= -2

        if self.verbose:
            print("Finished building XY axis class matrices.")

    def _build_xy_axis_heatmaps(self) -> None:
        """
        Builds the XY axis class heatmaps based on the xy_axis_matrix.

        Args:
            None
        
        Returns:
            None
        
        Raises:
            AssertionError: If xy_axis_matrix is None, not a torch.tensor,
            or has an invalid shape.
        """
        assert self.xy_axis_matrix is not None
        assert isinstance(self.xy_axis_matrix, torch.Tensor)
        assert len(self.xy_axis_matrix.shape) == 3

        if self.verbose:
            print("Building XY axis heatmaps...")

        self.xy_axis_heatmap = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )
    
        self.xy_axis_occupied_heatmap = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )

        self.xy_axis_empty_heatmap = torch.zeros(
            (self.xy_axis_matrix.shape[0], self.xy_axis_matrix.shape[1]),
            device=self.device
        )

        if self.verbose:
            print("Finished building XY axis heatmaps.")

    def _build_xy_axis_linspace(self) -> None:
        """
        Build the x and y axis linspace for the XY axis.

        This method calculates the x and y axis linspace based on the world
        bounds and axis resolution. It also extracts the horizontal and
        vertical boundaries, as well as the centers, from the axis linspace.
        Finally, it plots the quadrant boundaries, quadrant centers, and voxels
        for the XY axis.

        Args:
            None

        Returns:
            None
            
        Raises:
            NotImplementedError: If the environment dimensionality != 2.
        """

        if self.verbose:
            print("Building XY axis linspace...")

        if self.environment_dimensionality != 2:
            raise NotImplementedError

        assert self.world_bounds_norm[0] / self.axis_resolution == \
            int(self.world_bounds_norm[0] / self.axis_resolution)
        assert self.world_bounds_norm[1] / self.axis_resolution == \
            int(self.world_bounds_norm[1] / self.axis_resolution)
        
        xal_steps: int = int(self.world_bounds_norm[0] / self.axis_resolution)
        yal_steps: int = int(self.world_bounds_norm[1] / self.axis_resolution)
        
        xa = torch.linspace(
            start=0,
            end=self.world_bounds_norm[0],
            steps=(2 * xal_steps + 1),
            device=self.device
        )
        ya = torch.linspace(
            start=0,
            end=self.world_bounds_norm[1],
            steps=(2 * yal_steps + 1),
            device=self.device
        )

        # extract the horizontal and vertical boundaries from the axis linspace
        xab = xa[::2]
        yab = ya[::2]

        assert len(xab.shape) == 1
        assert len(yab.shape) == 1
        assert torch.min(xab) == 0
        assert torch.min(yab) == 0
        assert torch.max(xab) == self.world_bounds_norm[0]
        assert torch.max(yab) == self.world_bounds_norm[1]
    
        # extract the centers from the axis linspace
        xac = xa[1::2]
        yac = ya[1::2]

        assert len(xac.shape) == 1
        assert len(yac.shape) == 1
        assert torch.min(xac) == self.axis_resolution / 2
        assert torch.min(yac) == self.axis_resolution / 2
        assert torch.max(xac) == self.world_bounds_norm[0] - self.axis_resolution / 2
        assert torch.max(yac) == self.world_bounds_norm[1] - self.axis_resolution / 2

        self.xy_axis_linspace = (xac, yac)

        if self.verbose:
            print("Finished building XY axis linspace.")

        if self.plotting_flags["plot_xy_voxels"]:
            if self.verbose:
                print("Plotting XY axis boundaries, centers, and voxels...")

            vbp_sp: str = os.path.join(
                self.log_dir,
                "xy_voxel_boundaries.png"
            )
            spp.plot_quadrant_boundaries(
                qb_x=xab,
                qb_y=yab,
                world_bounds_norm=self.world_bounds_norm,
                save_path=vbp_sp,
                title_header="XY Axis Voxel Boundaries",
            )

            vcmg = torch.meshgrid(xac, yac, indexing="xy")
            vcmg = torch.stack(vcmg, dim=2)
            vcmg = vcmg.reshape((vcmg.shape[0] * vcmg.shape[1], 2))
            vcmg = vcmg.to(self.device)

            vcp_sp: str = os.path.join(
                self.log_dir,
                "xy_voxel_centers.png"
            )
            spp.plot_quadrant_centers(
                qcs=vcmg,
                world_bounds_norm=self.world_bounds_norm,
                save_path=vcp_sp,
                title_header="XY Axis Voxel Centers",
            )

            vp_sp: str = os.path.join(
                self.log_dir,
                "xy_voxels.png"
            )
            spp.plot_quadrants_and_centers(
                qcs=vcmg,
                qb_x=xab,
                qb_y=yab,
                world_bounds_norm=self.world_bounds_norm,
                save_path=vp_sp,
                title_header="XY Axis Voxels",
            )

            if self.verbose:
                print("Finished plotting XY axis boundaries, centers, and voxels.")


    def _build_xy_axis_vectors(self) -> None:
        """
        Builds the XY axis vectors using the SSP generator.

        This method generates the XY axis vectors based on the environment
        dimensionality using the SSP generator. It ensures that the generated
        vectors have the correct shape and dimensions.

        Args:
            None
        
        Returns:
            None

        Raises:
            AssertionError: If the generated vectors have an incorrect shape
            or dimension.
        """

        if self.verbose:
            print("Building XY axis vectors...")

        self.xy_axis_vectors = self.ssp_generator.generate(
            self.environment_dimensionality
        )

        axis_vector_sp: str = os.path.join(
            self.log_dir,
            "xy_axis_vectors.npy"
        )
        with open(axis_vector_sp, "wb") as f:
            np.save(f, self.xy_axis_vectors.detach().cpu().numpy())

        assert len(self.xy_axis_vectors.shape) == 2
        assert self.xy_axis_vectors.shape[0] == self.environment_dimensionality
        assert self.xy_axis_vectors.shape[1] == self.vsa_dimensions

        if self.verbose:
            print("Finished building XY axis vectors.")
    
    def _build_xy_axis_matrix(self) -> None:
        """
        Build the XY axis matrix using the xy_axis_linspace and xy_axis_vectors.

        This method constructs a matrix representing the XY axis by iterating
        over the xy_axis_linspace and xy_axis_vectors. For each combination of
        x and y values, it calculates the corresponding vector using the power
        function and binds them together using fractional binding.

        Returns:
            None
        """

        if self.verbose:
            print("Building XY axis matrix...")

        x_shape: tuple = self.xy_axis_linspace[0].shape[0]
        y_shape: tuple = self.xy_axis_linspace[1].shape[0]

        self.xy_axis_matrix = torch.zeros(
            (x_shape, y_shape, self.vsa_dimensions),
            device=self.device
        )

        if self.verbose:
            x_list = tqdm(
                self.xy_axis_linspace[0],
                desc="X Linspace",
            )
            y_list = tqdm(
                self.xy_axis_linspace[1],
                desc="Y Linspace",
                leave=False,
            )
        else:
            x_list = self.xy_axis_linspace[0]
            y_list = self.xy_axis_linspace[1]

        for i, x in enumerate(x_list):
            for j, y in enumerate(y_list):
                vs: list[torch.tensor] = [
                    spf.power(self.xy_axis_vectors[0], x, self.length_scale),
                    spf.power(self.xy_axis_vectors[1], y, self.length_scale)
                ]
                self.xy_axis_matrix[i, j, :] = spf.bind(vs, self.device)
        
        axis_matrix_sp: str = os.path.join(
            self.log_dir,
            "xy_axis_matrix.npy"
        )
        if self.save_axis_matrix:
            with open(axis_matrix_sp, "wb") as f:
                np.save(f, self.xy_axis_matrix.detach().cpu().numpy())

        if self.verbose:
            print("Finished building XY axis matrix.")

    def _build_quadrant_indices(self) -> None:
        """
        Builds the quadrant indices based on the quadrant axis bounds and axis
        resolution. The quadrant indices are calculated by dividing the
        quadrant axis bounds by the axis resolution.
        """

        if self.verbose:
            print("Building quadrant indices...")

        quadrant_indices_x = self.quadrant_axis_bounds[0][0] / self.axis_resolution
        quadrant_indices_y = self.quadrant_axis_bounds[0][1] / self.axis_resolution

        self.quadrant_indices_x = quadrant_indices_x.to(torch.int)
        self.quadrant_indices_y = quadrant_indices_y.to(torch.int)

        qix_sp: str = os.path.join(
            self.log_dir,
            "quadrant_indices_x.npy"
        )
        with open(qix_sp, "wb") as f:
            np.save(f, self.quadrant_indices_x.detach().cpu().numpy())
        
        qiy_sp: str = os.path.join(
            self.log_dir,
            "quadrant_indices_y.npy"
        )
        with open(qiy_sp, "wb") as f:
            np.save(f, self.quadrant_indices_y.detach().cpu().numpy())

        if self.verbose:
            print("Finished building quadrant indices.")

    def _build_quadrant_memory_hierarchy(self) -> None:
        """
        Builds the memory hierarchy for the quadrants.

        This method constructs the memory hierarchy for the quadrants based on
        the specified VSA dimensionality. It initializes the quadrant memory
        vectors as torch tensors with zeros.

        Args:
            None
        
        Returns:
            None

        Raises:
            NotImplementedError: If the hierarchy has more than one level.
        """

        if self.verbose:
            print("Building quadrant memory hierarchy...")

        if len(self.quadrant_hierarchy) > 1:
            raise NotImplementedError

        # Shape = {
        #   0 = number of quadrants by number of quadrants flattened,
        #   1 = number of dimensions in the vsa
        # }
        self.occupied_quadrant_memory_vectors = torch.zeros(
            size=(
                self.quadrant_hierarchy[0] ** self.environment_dimensionality,
                self.vsa_dimensions
            ),
            device=self.device
        )
        self.empty_quadrant_memory_vectors = torch.clone(self.occupied_quadrant_memory_vectors)

        if self.verbose:
            print("Finished building quadrant memory hierarchy.")

    def _build_quadrant_hierarchy(self) -> None:
        """
        Builds the quadrant hierarchy.

        This method builds the quadrant hierarchy based on the specified sizes
        in the `quadrant_hierarchy` list. Each level of the hierarchy is built
        using the `build_quadrant_level` method.
                
        Args:
            None
        
        Returns:
            None

        Raises:
            AssertionError: If the `quadrant_hierarchy` list is empty or if the
                first element is not an integer or is less than or equal to 0.
        """

        if self.verbose:
            print("Building quadrant hierarchy...")
        
        assert len(self.quadrant_hierarchy) == 1
        assert isinstance(self.quadrant_hierarchy[0], int)
        assert self.quadrant_hierarchy[0] > 0

        if self.verbose:
            iterator = tqdm(
                self.quadrant_hierarchy,
                desc="Building Quadrant Hierarchy",
                total=len(self.quadrant_hierarchy),
            )
        else:
            iterator = self.quadrant_hierarchy

        for level, size in enumerate(iterator):
            self.build_quadrant_level(level, size)

        if self.verbose:
            print("Finished building quadrant hierarchy.")

    def build_quadrant_level(self, level: int, size: int) -> None:
        """
        Build the quadrant level based on the given level and size.

        Args:
            level (int): The level of the quadrant.
            size (int): The size of the quadrant.

        Returns:
            None

        Raises:
            NotImplementedError: If the level is not 0.
        """

        if self.verbose:
            print(f"Building quadrant level [{level}]...")

        # Verify that the world bounds are divisible by the quadrant size
        assert self.world_bounds_norm[0] % size == 0
        assert self.world_bounds_norm[1] % size == 0
        assert self.world_bounds_norm[0] / self.axis_resolution % size == 0
        assert self.world_bounds_norm[1] / self.axis_resolution % size == 0
        
        # Calculate the quadrant size across x and y axes in meters
        # while also taking into account the level of the quadrant
        if level == 0:
            size_x_meters: float = self.world_bounds_norm[0] / size
            size_y_meters: float = self.world_bounds_norm[1] / size
        
        size_x_bins = int(size_x_meters / self.axis_resolution)
        size_y_bins = int(size_y_meters / self.axis_resolution)

        if self.verbose:
            print(f"Quadrant Level [{level}] - Number of Quadrants: {size ** 2}")
            print(f"Quadrant Level [{level}] - Quadrant Size (Bins): {size_x_bins}, {size_y_bins}")
            print(f"Quadrant Level [{level}] - Quadrant Size (Meters): {size_x_meters}, {size_y_meters}")
            print(f"Quadrant Level [{level}] - Level Size (Bins): {size_x_bins * size}, {size_y_bins * size}")
            print(f"Quadrant Level [{level}] - Level Size (Meters): {size_x_meters * size}, {size_y_meters * size}")

        # Verify that the level size calculated based on all quadrant sizes
        if level == 0:
            assert size_x_meters * size == self.world_bounds_norm[0]
            assert size_y_meters * size == self.world_bounds_norm[1]
        else:
            raise NotImplementedError

        # calculate the vertical and horizontal bounds of the quadrants
        qb_x = torch.linspace(0, self.world_bounds_norm[0], size + 1)
        qb_y = torch.linspace(0, self.world_bounds_norm[1], size + 1)

        assert len(qb_x.shape) == 1
        assert len(qb_y.shape) == 1
        assert qb_x.shape[0] == size + 1
        assert qb_y.shape[0] == size + 1
        assert torch.min(qb_x) == 0
        assert torch.min(qb_y) == 0
        assert torch.max(qb_x) == self.world_bounds_norm[0]
        assert torch.max(qb_y) == self.world_bounds_norm[1]

        qbp_sp: str = os.path.join(
            self.log_dir,
            f"quadrant_boundaries_level_{level}.png"
        )
        spp.plot_quadrant_boundaries(
            qb_x=qb_x,
            qb_y=qb_y,
            world_bounds_norm=self.world_bounds_norm,
            quadrant_level=level,
            save_path=qbp_sp
        )
            
        # calculate the quadrant centers
        qcs_x = torch.linspace(0, self.world_bounds_norm[0], 2 * size + 1)[1::2]
        qcs_y = torch.linspace(0, self.world_bounds_norm[1], 2 * size + 1)[1::2]

        assert len(qcs_x.shape) == 1
        assert len(qcs_y.shape) == 1
        assert qcs_x.shape[0] == size
        assert qcs_y.shape[0] == size
        assert torch.min(qcs_x) == self.world_bounds_norm[0] / size / 2
        assert torch.min(qcs_y) == self.world_bounds_norm[1] / size / 2
        assert torch.max(qcs_x) == self.world_bounds_norm[0] - self.world_bounds_norm[0] / size / 2
        assert torch.max(qcs_y) == self.world_bounds_norm[1] - self.world_bounds_norm[1] / size / 2

        self.quadrant_axis_bounds.append((qb_x, qb_y))
        
        qcmg = torch.meshgrid(qcs_x, qcs_y, indexing="xy")
        qcs = torch.stack(qcmg, dim=2)
        qcs = qcs.reshape((size ** 2, 2))
        qcs = qcs.to(self.device)

        assert len(qcs.shape) == 2
        assert qcs.shape[0] == size ** 2
        assert qcs.shape[1] == self.environment_dimensionality
        assert torch.min(qcs[:, 0]) == self.world_bounds_norm[0] / size / 2
        assert torch.min(qcs[:, 1]) == self.world_bounds_norm[1] / size / 2
        assert torch.max(qcs[:, 0]) == self.world_bounds_norm[0] - self.world_bounds_norm[0] / size / 2
        assert torch.max(qcs[:, 1]) == self.world_bounds_norm[1] - self.world_bounds_norm[1] / size / 2

        self.quadrant_centers.append(qcs)

        if self.verbose:
            print(f"Finished building quadrant level [{level}].")

        if self.verbose:
            print("Plotting quadrant centers and quadrants...")

        qcp_sp: str = os.path.join(
            self.log_dir,
            f"quadrant_centers_level_{level}.png"
        )
        spp.plot_quadrant_centers(
            qcs=qcs,
            world_bounds_norm=self.world_bounds_norm,
            quadrant_level=level,
            save_path=qcp_sp
        )

        qp_sp: str = os.path.join(
            self.log_dir,
            f"quadrants_level_{level}.png"
        )
        spp.plot_quadrants_and_centers(
            qcs=qcs,
            qb_x=qb_x,
            qb_y=qb_y,
            world_bounds_norm=self.world_bounds_norm,
            quadrant_level=level,
            save_path=qp_sp
        )

        if self.verbose:
            print("Finished plotting quadrant centers and quadrants.")
