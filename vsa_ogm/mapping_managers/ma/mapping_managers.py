import numpy as np
from omegaconf import DictConfig
from sklearn import metrics
from typing import List

from vsa_ogm.logging import BaseLogger
from vsa_ogm.mappers.sa import BaseSingleAgentMapper

VALID_METRICS: List[str] = ["auc", "f1", "precision", "recall", "accuracy", "nll"]

class MultiAgentMappingManager:
    """
    TODO Finish Documentation
    """

    config: DictConfig = None
    loggers: List[BaseLogger] = []
    num_agents: int = 0
    plotting_flags: DictConfig = None
    saving_flags: DictConfig = None
    print_header: str = None
    agent_configs: List[DictConfig] = []

    all_X_train: List[np.ndarray] = []
    all_y_train: List[np.ndarray] = []
    all_X_test: List[np.ndarray] = []
    all_y_test: List[np.ndarray] = []
    mapper: BaseSingleAgentMapper = None


    def __init__(self, config: DictConfig, loggers: List[BaseLogger],
                 print_header: str = "(Multi-Agent Mapping Manager)") -> None:
        """
        Initializes a MappingManager object.

        Args:
            config (DictConfig): The configuration object containing the
                mapping information.
        """
        self.config: DictConfig = config["agent_manager"]
        self.loggers: List[BaseLogger] = loggers
        self.print_header: str = print_header

        self.num_agents: int = config["num_agents"]
        self.agent_configs: List[DictConfig] = [config[f"agent_{i}"] for i in range(self.num_agents)]

        # check if all metrics are valid
        for agent_config in self.agent_configs:
            for metric in agent_config["metrics"]:
                if metric not in VALID_METRICS:
                    raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {VALID_METRICS}")

        # print config pretty
        print(config)

    def run(self) -> None:
        """
        Run the MappingManager.
        """

        pass


