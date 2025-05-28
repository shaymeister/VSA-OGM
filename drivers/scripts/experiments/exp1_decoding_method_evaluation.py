import itertools
import os
import subprocess
import sys
import tempfile
from omegaconf import DictConfig, OmegaConf

from vsa_ogm.data import load_data
from vsa_ogm.data.sa import BaseSingleAgentDataset
from vsa_ogm.logging import OGMLogger, WANBDLogger
from vsa_ogm.mapping_managers.sa import SingleAgentMappingManager
from vsa_ogm.utilities import (
    parse_args,
    validate_overrides,
    print_introduction
)


def launch_evaluation(base_config: DictConfig, override_config: DictConfig) -> None:
    """
    Launch the experiment.
    """

    print("override_config:")
    print(OmegaConf.to_yaml(override_config))

    # Merge the configurations
    config = OmegaConf.merge(base_config, override_config)

    # Initialize the loggers
    local_logger = OGMLogger(config)
    online_logger = WANBDLogger(config)
    loggers = [local_logger, online_logger]
    [logger.log_config(config) for logger in loggers]

    # Load the data
    dataset: BaseSingleAgentDataset = load_data(config, loggers)

    # create the mapping manager
    mapping_manager = SingleAgentMappingManager(config, loggers)

    # run the mapping manager
    mapping_manager.run(dataset)

    # close the loggers
    [logger.close() for logger in loggers]

def extract_lists(d: DictConfig, prefix: str = ""):
    """
    Recursively extract lists from a nested dictionary along with their hierarchical keys.
    
    Returns:
        keys (list of str): Full keys in "key1.key2.key3" format.
        values (list of lists): Corresponding lists of values.
    """
    result_keys, result_values = [], []

    for key, value in d.items():
        if OmegaConf.is_list(value):
            result_keys.append(f"{prefix}.{key}".lstrip("."))
            result_values.append(value)
        elif OmegaConf.is_dict(value):
            keys, values = extract_lists(value, f"{prefix}.{key}".lstrip("."))
            result_keys.extend(keys)
            result_values.extend(values)
                                 
    return result_keys, result_values


def main(config: DictConfig) -> None:
    """
    Main function to execute the script.

    Args:
        config (omegaconf.DictConfig): Merged configuration.
    """
    base_log_dir = config.logging.save_dir
    base_log_dir = os.path.join(base_log_dir, config.wandb.project_name)

    # Extract structured parameter lists
    keys, parameter_lists = extract_lists(config.experiment_parameters)

    # Compute Cartesian product
    combinations = list(itertools.product(*parameter_lists))

    # Save base config to a temp file once
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as base_file:
        OmegaConf.save(config=config, f=base_file.name)
        base_config_path = base_file.name

    for combo in combinations:
        # Create a dot string for each combination
        combo_dot_strings = [f"{key}={value}" for key, value in zip(keys, combo)]

        # create a name for the combination
        combo_name = "_".join([f"{key}={value}" for key, value in zip(keys, combo)])
        combo_name = combo_name.replace(".", "-")
        combo_log_dir = os.path.join(base_log_dir, combo_name)
        combo_dot_strings.append("logging.save_dir=" + combo_log_dir)
        combo_dot_strings.append("experiment_name=" + combo_name)

        print(combo_dot_strings)

        # Save override config for this combo
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as override_file:
            override_cfg = OmegaConf.from_dotlist(combo_dot_strings)
            OmegaConf.save(config=override_cfg, f=override_file.name)
            override_config_path = override_file.name

        print("Launching:", combo_dot_strings)
        # Use Popen to stream output live
        process = subprocess.Popen(
            [sys.executable, "drivers/scripts/experiments/launch_evaluation.py", base_config_path, override_config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # # Read and print output line-by-line as it's generated
        # with process.stdout:
        #     for line in iter(process.stdout.readline, ''):
        #         print(f"[{combo_name}] {line}", end='')

        process.wait()
        print(f"[{combo_name}] Completed with return code {process.returncode}")


if __name__ == "__main__":
    file_name: str = os.path.basename(__file__)
    description: str = """
        Evaluate the performance of different decoding methods across the
        configured dataset.
    """
    args, unknown_args = parse_args(
        prog=file_name,
        description=description
    )
    config = OmegaConf.load(args.config)
    override_config = OmegaConf.from_dotlist(unknown_args)
    validate_overrides(config, override_config)
    print_introduction(args, config, override_config)
    config = OmegaConf.merge(config, override_config)
    main(config)