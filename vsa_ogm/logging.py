import matplotlib.pyplot as plt
import numpy as np
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
    
    def log_string(self, string: str) -> None:
        """
        TODO Finish Documentation
        """
        raise NotImplementedError("log_string is not implemented in BaseLogger.")
    
    def close(self) -> None:
        """
        TODO Finish Documentation
        """
        raise NotImplementedError("close is not implemented in BaseLogger.")

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

    def log_string(self, string: str) -> None:
        """
        This methods is simply a pass-through for the print function because
        WandB will log all print statements automatically.
        """
        return
    
    def log_image(self, image: np.ndarray, caption: str) -> None:
        """
        Log an image to the WandB logger.

        Arguments:
        ----------
        image : np.ndarray
            The image to log.
        caption : str
            The caption for the image.
        """
        pass
    
    def close(self) -> None:
        """
        Close the WandB session.
        """
        pass

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

        std_out_fp: str = os.path.join(self.experiment_dir, "stdout.txt")
        self.std_out_file = open(std_out_fp, "w")

    def log_config(self, config: DictConfig) -> None:
        """
        Save the configuration to a file.

        Arguments:
        ----------
        config : DictConfig
            The configuration to save.
        """
        OmegaConf.save(config, os.path.join(self.experiment_dir, self.config_fname))

    def log_string(self, string: str) -> None:
        """
        Write a string to the stdout file and print it to the console.

        Arguments:
        ----------
        string : str
            The string to write to the file.
        """
        self.std_out_file.write(string + "\n")
        if self.verbose:
            print(f"{self.print_header} wrote string to file")

    def log_image(self, image: np.ndarray, caption: str) -> None:
        """
        Log an image to the logger.

        Arguments:
        ----------
        image : np.ndarray
            The image to log.
        caption : str
            The caption for the image.
        """
        plt.imshow(image)
        plt.title(caption)
        plt.savefig(os.path.join(self.experiment_dir, f"{caption}.png"))

    def close(self) -> None:
        """
        close all file handlers
        """
        self.std_out_file.close()
