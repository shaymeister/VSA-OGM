import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from typing import List

from vsa_ogm.data.sa import BaseSingleAgentDataset
from vsa_ogm.logging import BaseLogger
from vsa_ogm.mappers.sa import BaseSingleAgentMapper

VALID_METRICS: List[str] = ["auc", "f1", "precision", "recall", "accuracy", "nll"]

class SingleAgentMappingManager:
    """
    TODO Finish Documentation
    """

    all_X_train: List[np.ndarray] = []
    all_y_train: List[np.ndarray] = []
    all_X_test: List[np.ndarray] = []
    all_y_test: List[np.ndarray] = []
    mapper: BaseSingleAgentMapper = None

    def __init__(self, config: DictConfig, loggers: List[BaseLogger],
                 print_header: str = "(Mapping Manager)") -> None:
        """
        Initializes a MappingManager object.

        Args:
            config (DictConfig): The configuration object containing the
                mapping information.
        """
        self.config: DictConfig = config
        self.loggers: List[BaseLogger] = loggers
        self.print_header: str = print_header
        self.seed: int = config.mapping_manager.seed
        self.test_size: float = config.data.test_split
        self.verbose: bool = config.mapping_manager.verbose

        # extract parameters from the config
        self.metrics: List[str] = config.mapping_manager.metrics

        # check if the metrics are valid
        for metric in self.metrics:
            if metric not in VALID_METRICS:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {VALID_METRICS}")
            
        self.plotting_flags: DictConfig = config.mapping_manager.plotting_flags

        self._initialize_mapper()

    def run(self, dataset: BaseSingleAgentDataset) -> None:
        """
        Run the MappingManager on the given dataset.

        Args:
            dataset (BaseSingleAgentDataset): The dataset to run the
                MappingManager on.
        """

        dataset_length: int = len(dataset)

        # log the start of the mapping manager
        string = f"{self.print_header} Running Mapping Manager on dataset with {dataset_length} steps."
        if self.verbose:
            print(string)
        for logger in self.loggers:
            logger.log_string(string)

        # iterate through the dataset
        for idx in range(dataset_length):
            # initialize per iteration variables
            complete_metric_dict: dict = {}
            complete_images_dict: dict = {}
            complete_figures_dict: dict = {}

            # log the current step
            string = f"{self.print_header} Processing dataset step {idx}."
            if self.verbose:
                print(string)
            for logger in self.loggers:
                logger.log_string(string)

            # get the data at the current index
            data_batch: dict = dataset[idx]
            X: np.ndarray = data_batch["lidar_data"]
            y: np.ndarray = data_batch["occupancy"]

            # split the data into stratified training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.seed,
                stratify=y
            )

            # store the training and testing data
            self.all_X_train.append(X_train)
            self.all_y_train.append(y_train)
            self.all_X_test.append(X_test)
            self.all_y_test.append(y_test)

            # run the mapper on the data
            fit_metrics: dict = self.mapper.fit(X_train, y_train)
            complete_metric_dict.update(fit_metrics)

            # predict the testing data
            y_pred, pred_metrics = self.mapper.predict(X_test)
            complete_metric_dict.update(pred_metrics)

            

    def _initialize_mapper(self) -> None:
        """
        Initialize the mapper based on the configuration.
        """
        mapper_type: str = self.config.mapping.mapping_type
        if mapper_type == "SA_VSA_OGM":
            from vsa_ogm.mappers.sa.sa_vsa_mapper import SA_VSA_OGM
            self.mapper = SA_VSA_OGM(self.config, self.loggers)
        else:
            raise ValueError(f"Invalid mapper type: {mapper_type}.")


