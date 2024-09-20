from omegaconf import DictConfig

from vsa_ogm.dataloaders import (
    CSVDataLoader,
    PickleDataLoader,
    ToySimDataLoader,
)


def load_single_data(config: DictConfig) -> tuple:
    """
    Load a single data set based on the provided configuration.

    Args:
        config (DictConfig): The configuration object containing the
            dataset information.

    Returns:
        tuple: A tuple containing the dataloader object and the world size.

    Raises:
        ValueError: If the dataset name is unknown.
    """

    if config.verbose:
        print("Loading Data...")

    dataloader = None
    world_size = None

    if config.data.dataset_name == "toysim":
        dataloader = ToySimDataLoader(config.data.toysim)
        world_size = config.data.toysim.world_bounds
    elif config.data.dataset_name == "intel":
        dataloader = CSVDataLoader(config.data.intel)
        world_size = config.data.intel.world_bounds
    else:
        raise ValueError(F"Unknown dataset: {config.data.dataset_name}")

    return dataloader, world_size


def load_fusion_data(config: DictConfig) -> tuple:
    """
    Load fusion data based on the provided configuration.

    Args:
        config (DictConfig): The configuration for loading the fusion data.

    Returns:
        tuple: A tuple containing the dataloaders and the world size.

    Raises:
        None
    """

    if config.verbose:
        print("Loading Data...")

    dataloaders = []
    world_size = None

    if config.data.dataset_name == "toysim":
        for agent in config.data.toysim:
            dataloader = ToySimDataLoader(config.data.toysim[agent])
            dataloaders.append(dataloader)
            world_size = config.data.toysim[agent].world_bounds

    elif config.data.dataset_name == "intel":
        # for agent in config.data.intel:
        dataloader = PickleDataLoader(config.data.intel)
        dataloaders.append(dataloader)
        world_size = config.data.intel.world_bounds

    return dataloader, world_size