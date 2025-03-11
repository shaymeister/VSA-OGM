import itertools
import os
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


def main(config: DictConfig) -> None:
    """
    Main function to execute the script.

    Args:
        config (omegaconf.DictConfig): Merged configuration.
    """

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