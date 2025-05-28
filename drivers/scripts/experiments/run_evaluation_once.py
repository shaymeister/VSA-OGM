import os
import sys
from omegaconf import OmegaConf
from vsa_ogm.data import load_data
from vsa_ogm.data.sa import BaseSingleAgentDataset
from vsa_ogm.logging import OGMLogger, WANBDLogger
from vsa_ogm.mapping_managers.sa import SingleAgentMappingManager
from vsa_ogm.utilities import parse_args, validate_overrides, print_introduction


def cast_override_types(cfg):
    """Cast known string override values to int/float types."""
    type_overrides = {
        "mapping.axis_resolution": float,
        "mapping.vector_dimensionality": int,
        "mapping.vector_length_scale": float,
        "mapping.num_tiles": int,
    }

def launch_evaluation(config):
    loggers = [OGMLogger(config), WANBDLogger(config)]
    [logger.log_config(config) for logger in loggers]

    dataset: BaseSingleAgentDataset = load_data(config, loggers)
    mapping_manager = SingleAgentMappingManager(config, loggers)
    mapping_manager.run(dataset)

    [logger.close() for logger in loggers]

if __name__ == "__main__":
    file_name: str = os.path.basename(__file__)
    description: str = "Single evaluation run for parameter sweep."

    args, unknown_args = parse_args(prog=file_name, description=description)
    base_config = OmegaConf.load(args.config)
    overrides = OmegaConf.from_dotlist(unknown_args)

    type_overrides = {
        "mapping.axis_resolution": float,
        "mapping.vector_dimensionality": int,
        "mapping.vector_length_scale": float,
        "mapping.num_tiles": int,
    }

    for key, cast_type in type_overrides.items():
        if key in overrides:
            overrides[key] = cast_type(overrides[key])

    validate_overrides(base_config, overrides)

    print(f"Overrides: {overrides}")
    
    config = OmegaConf.merge(base_config, overrides)
    print(f"Configuration: {OmegaConf.to_yaml(config)}")

    launch_evaluation(config)
