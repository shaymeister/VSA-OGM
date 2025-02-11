from omegaconf import DictConfig
from typing import List

from vsa_ogm.data.sa import BaseSingleAgentDataset
from vsa_ogm.logging import BaseLogger

VALID_METRICS: List[str] = ["auc", "f1", "precision", "recall", "accuracy", "nll"]

class SingleAgentMappingManager:
    """
    TODO Finish Documentation
    """
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
        self.verbose: bool = config.mapping_manager.verbose

        # extract parameters from the config
        self.metrics: List[str] = config.mapping_manager.metrics

        # check if the metrics are valid
        for metric in self.metrics:
            if metric not in VALID_METRICS:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {VALID_METRICS}")
            
        self.plotting_flags: DictConfig = config.mapping_manager.plotting_flags

    def run(self, dataset: BaseSingleAgentDataset) -> None:
        """
        Run the MappingManager on the given dataset.

        Args:
            dataset (BaseSingleAgentDataset): The dataset to run the
                MappingManager on.
        """

        dataset_length: int = len(dataset)

        string = f"{self.print_header} Running Mapping Manager on dataset with {dataset_length} steps."
        if self.verbose:
            print(string)
        for logger in self.loggers:
            logger.log_string(string)
