from omegaconf import DictConfig
from typing import List, Tuple

from vsa_ogm.logging import BaseLogger

VALID_DATASETS: List[str] = ["toysim", "ablation", "intel"]
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

    dataset_name = config.data.dataset_name
    exp_type: str = config.experiment_type

    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset - {dataset_name} - must be in [{VALID_DATASETS}]")
    
    if exp_type not in VALID_EXPERIMENT_TYPES:
        raise ValueError(f"Unknown experiment type -  {exp_type} - must be in [{VALID_EXPERIMENT_TYPES}]")
    
    if exp_type == "multi-agent":
        raise NotImplementedError("Multi-agent experiments are not yet supported.")
    
    if dataset_name == "toysim" and exp_type == "single_agent":
        from vsa_ogm.data.sa.toysim_sa_dataset import ToySimSingleAgentDataset
        dataset = ToySimSingleAgentDataset(config, loggers)
    elif dataset_name == "ablation" and exp_type == "single_agent":
        from vsa_ogm.data.sa.ablation_sa_dataset import AblationSingleAgentDataset
        dataset = AblationSingleAgentDataset(config, loggers)
    elif dataset_name == "intel" and exp_type == "single_agent":
        from vsa_ogm.data.sa.intel_sa_dataset import IntelSingleAgentDataset
        dataset = IntelSingleAgentDataset(config, loggers)
    else:
        raise ValueError(f"Unknown dataset - {dataset_name} - must be in [{VALID_DATASETS}]")

    return dataset

