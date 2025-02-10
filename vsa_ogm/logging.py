from omegaconf import DictConfig
import os

class WANBDLogger:
    """
    TODO Finish Documentation
    """
    def __init__(self, config: DictConfig, print_header: str = "(WandB Logger)") -> None:
        """
        TODO Finish Documentation
        """
        self.config = config

        self.print_header: str = print_header
        self.project_name: str = config.wandb.project_name
        self.experiment_name: str = config.experiment_name
        self.notes: str = config.wandb.notes
        self.verbose: bool = config.wandb.verbose


class OGMLogger:
    """
    TODO Finish Documentation
    """

    def __init__(self, config: DictConfig, print_header: str = "(Local Logger)") -> None:
        """
        TODO Finish Documentation
        """
        self.config = config

        self.print_header: str = print_header
        self.override_existing: bool = config.logging.override_existing
        self.save_dir: str = config.logging.save_dir
        self.verbose: bool = config.logging.verbose

        # initialize logger directory
        if os.path.exists(self.save_dir) and not self.override_existing:
            # ask the user if they would like to override
            answer: str = input(f"{print_header} Directory {self.save_dir} already exists. Would you like to overwrite? (y/n): ")
            if answer.lower() != "y":
                raise ValueError(f"{print_header} Directory {self.save_dir} already exists. Set logging_override_existing=True to overwrite.")
            
        if os.path.exists(self.save_dir):
            os.rmdir(self.save_dir)
        
        os.makedirs(self.save_dir)

    





    