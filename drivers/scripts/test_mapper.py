import argparse
from typing import List, Dict, Any
import omegaconf
import os
from tabulate import tabulate

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the input config file")
    parser.add_argument("--kwargs", nargs='*', help="Additional keyword arguments to override config values")
    return parser.parse_args()

def parse_kwargs(kwargs_list: List[str]) -> Dict[str, Any]:
    """
    Parse the list of keyword arguments into a dictionary.

    Args:
        kwargs_list (List[str]): List of keyword arguments in the form of key=value.

    Returns:
        Dict[str, Any]: Dictionary of parsed keyword arguments.
    """
    kwargs_dict = {}
    if kwargs_list:
        for item in kwargs_list:
            key, value = item.split('=')
            kwargs_dict[key] = value
    return kwargs_dict

def validate_kwargs(kwargs: Dict[str, Any], config: omegaconf.DictConfig) -> None:
    """
    Validate that the provided keyword arguments are valid keys in the config.

    Args:
        kwargs (Dict[str, Any]): Dictionary of keyword arguments.
        config (omegaconf.DictConfig): Baseline configuration.

    Raises:
        ValueError: If any keyword argument is not a valid key in the config.
    """
    invalid_keys = [key for key in kwargs if key not in config]
    if invalid_keys:
        raise ValueError(f"Invalid config keys: {', '.join(invalid_keys)}")

def print_introduction(config: omegaconf.DictConfig, kwargs: Dict[str, Any], delimiter: str = "*", delimiter_width: int = 80) -> None:

    """
    Print an introduction with the default config and overridden values.

    Args:
        config (omegaconf.DictConfig): Baseline configuration.
        kwargs (Dict[str, Any]): Dictionary of overridden keyword arguments.
        delimiter (str, optional): Delimiter character for the printed output. Defaults to "*".
        delimiter_width (int, optional): Width of the delimiter line. Defaults to 80.
    Returns:
        None
    
    """
    print(delimiter * delimiter_width)
    print(f"\nConfig file: {args.config}\n")
    print("Default Config:")
    default_config_table = [[key, value] for key, value in config.items()]
    print(tabulate(default_config_table, headers=["Key", "Value"], tablefmt="grid"))

    if kwargs:
        print("\nOverridden Config:")
        overridden_values_table = [[key, config[key], value] for key, value in kwargs.items()]
        print(tabulate(overridden_values_table, headers=["Key", "Original Value", "New Value"], tablefmt="grid"))
    
    print()
    print(delimiter * delimiter_width)

def main(config: omegaconf.DictConfig) -> None:
    """
    Main function to execute the script.

    Args:
        config (omegaconf.DictConfig): Merged configuration.
    """
    pass

if __name__ == "__main__":
    args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    kwargs = parse_kwargs(args.kwargs)
    validate_kwargs(kwargs, config)
    print_introduction(config, kwargs)
    config = omegaconf.OmegaConf.merge(config, kwargs)
    main(config)