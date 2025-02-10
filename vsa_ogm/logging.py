from omegaconf import DictConfig, OmegaConf
import os
import shutil

class BaseLogger:
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, print_header: str = "(Base Logger)") -> None:
        """
        TODO Finish Documentation
        """
        self.config = config
        self.experiment_name: str = config.experiment_name
        self.print_header: str = print_header

    def log_config(self, config: DictConfig) -> None:
        """
        TODO Finish Documentation
        """
        raise NotImplementedError("log_config is not implemented in BaseLogger.")

class WANBDLogger(BaseLogger):
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, print_header: str = "(WandB Logger)") -> None:
        """
        TODO Finish Documentation
        """
        super().__init__(config, print_header)
        self.config = config

        self.project_name: str = config.wandb.project_name
        self.experiment_name: str = config.experiment_name
        self.notes: str = config.wandb.notes
        self.verbose: bool = config.wandb.verbose

    def log_config(self, config: DictConfig) -> None:
        """
        TODO Finish Documentation
        """
        print(f"{self.print_header} (TODO) Logging Configuration")


class OGMLogger(BaseLogger):
    """
    TODO Finish Documentation
    """

    def __init__(self, config: DictConfig, config_fname: str = "config.yaml",
                 print_header: str = "(Local Logger)") -> None:
        """
        TODO Finish Documentation
        """
        super().__init__(config, print_header)
        self.config_fname: str = config_fname
        self.experiment_name: str = config.experiment_name
        self.override_existing: bool = config.logging.override_existing
        self.save_dir: str = config.logging.save_dir
        self.verbose: bool = config.logging.verbose

        self.experiment_dir: str = os.path.join(self.save_dir, self.experiment_name)

        # initialize logger directory
        if os.path.exists(self.experiment_dir) and not self.override_existing:
            # ask the user if they would like to override
            answer: str = input(f"{print_header} Directory {self.experiment_dir} already exists. Would you like to overwrite? (y/n): ")
            if answer.lower() != "y":
                raise ValueError(f"{print_header} Directory {self.experiment_dir} already exists. Set logging_override_existing=True to overwrite.")
            
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)
        
        os.makedirs(self.experiment_dir)

    def log_config(self, config: DictConfig) -> None:
        """
        TODO Finish Documentation
        """
        OmegaConf.save(config, os.path.join(self.experiment_dir, self.config_fname))

    





    