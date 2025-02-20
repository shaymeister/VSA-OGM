import argparse
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf
import sys
from tabulate import tabulate
from typing import Any, Dict, List, Tuple

def train_test_split(data: dict, test_split: float) -> tuple:
    """
    Split the data into training and testing sets.

    Args:
        data (dict): A dictionary containing the lidar data and occupancy data.
        test_split (float): The percentage of the data used for testing.

    Returns:
        tuple: A tuple containing the training lidar data, training occupancy
            data, testing lidar data, and testing occupancy data.
    """

    assert isinstance(data, dict)
    assert "lidar_data" in data.keys()
    assert "occupancy" in data.keys()
    assert isinstance(data["lidar_data"], np.ndarray)
    assert isinstance(data["occupancy"], np.ndarray)
    assert data["lidar_data"].shape[0] == data["occupancy"].shape[0]
    assert isinstance(test_split, float)
    assert test_split > 0.0 and test_split < 1.0

    data_indices: np.ndarray = np.arange(data["lidar_data"].shape[0])
    np.random.shuffle(data_indices)

    test_length: int = int(test_split * data_indices.shape[0])
    
    test_indices: np.ndarray = data_indices[:test_length]
    train_indices: np.ndarray = data_indices[test_length:]

    test_lidar: np.ndarray = data["lidar_data"][test_indices]
    test_occupancy: np.ndarray = data["occupancy"][test_indices]

    train_lidar: np.ndarray = data["lidar_data"][train_indices]
    train_occupancy: np.ndarray = data["occupancy"][train_indices]

    return train_lidar, train_occupancy, test_lidar, test_occupancy


def parse_args(**kwargs) -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
        List[str]: List of unknown arguments.
    """
    if kwargs is not None:
        parser = argparse.ArgumentParser(**kwargs)
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the input config file")
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def validate_overrides(loaded_config: DictConfig, argued_config: DictConfig) -> DictConfig:
    """
    Parse the list of keyword arguments into a dictionary.

    Args:
        kwargs_list (List[str]): List of keyword arguments in the form of key=value.

    Returns:
        Dict[str, Any]: Dictionary of parsed keyword arguments.
    """
    # Find unknown keys in the CLI configuration
    unknown_keys = [key for key in argued_config if key not in loaded_config]

    if unknown_keys:
        print(f"Error: Unknown keys in command-line arguments: {unknown_keys}")
        sys.exit(1)  # Exit with an error code

    # Merge the CLI configuration into the base configuration
    merged_cfg = OmegaConf.merge(loaded_config, argued_config)
    return merged_cfg


def print_introduction(args:argparse.Namespace, config: omegaconf.DictConfig,
        kwargs: Dict[str, Any], delimiter: str = "-",
        delimiter_width: int = 80) -> None:

    """
    Print an introduction with the default config and overridden values.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        config (omegaconf.DictConfig): Baseline configuration.
        kwargs (Dict[str, Any]): Dictionary of overridden keyword arguments.
        delimiter (str, optional): Delimiter character for the printed output. Defaults to "-".
        delimiter_width (int, optional): Width of the delimiter line. Defaults to 80.
    Returns:
        None
    
    """
    print(delimiter * delimiter_width)
    print(f"\nConfig file: {args.config}\n")
    print("Default Config:")
    default_config_table = []
    # Print each level of the configuration
    for key, value in config.items():
        if isinstance(value, DictConfig):
            for subkey, subvalue in value.items():
                default_config_table.append([f"{key}.{subkey}", subvalue])
        else:
            default_config_table.append([key, value])
    print(tabulate(default_config_table, headers=["Key", "Value"], tablefmt="grid"))

    if kwargs:
        print("\nOverridden Config:")

        overridden_config_table = []

        for key, value in kwargs.items():
            if isinstance(value, DictConfig):
                for subkey, subvalue in value.items():
                    overridden_config_table.append([f"{key}.{subkey}", config[key][subkey], subvalue])
            else:
                overridden_config_table.append([key, config[key], value])

        # overridden_values_table = [[key, config[key], value] for key, value in kwargs.items()]
        print(tabulate(overridden_config_table, headers=["Key", "Original Value", "New Value"], tablefmt="grid"))
    
    print()
    print(delimiter * delimiter_width)