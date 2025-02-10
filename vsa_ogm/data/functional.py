from omegaconf import DictConfig
from typing import List, Tuple

from vsa_ogm.logging import BaseLogger

VALID_DATASETS: List[str] = ["toysim"]
VALID_EXPERIMENT_TYPES: List[str] = ["single_agent", "multi_agent"]

def load_data(config: DictConfig, loggers: List[BaseLogger]):
    """
    Load the data based on the provided configuration.

    Args:
        config (DictConfig): The configuration object containing the
            dataset information.

    Returns:
        tuple: A tuple containing the dataloader object and the world size.

    Raises:
        ValueError: If the dataset name is unknown.
    """

    dataset_name = config.dataset.name

    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if config.experiment_type not in VALID_EXPERIMENT_TYPES:
        raise ValueError(f"Unknown experiment type: {config.experiment_type}")
    
    if config.experiment_type == "multi-agent":
        raise NotImplementedError("Multi-agent experiments are not yet supported.")
    
    if dataset_name == "toysim" and config.experiment_type == "single_agent":
        from vsa_ogm.data.single_agent.toysim_single_agent_dataset import ToySimSingleAgentDataset
        dataset = ToySimSingleAgentDataset(config, loggers)

    return dataset

