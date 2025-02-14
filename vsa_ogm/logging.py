import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import shutil
from typing import Dict

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
    
    def log_image(self, image: np.ndarray, caption: str, epoch: int = -1) -> None:
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

    def log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Log metrics to the WandB logger.

        Arguments:
        ----------
        metrics : Dict[str, float]
            The metrics to log.
        epoch : int
            The epoch number.
        """
        pass

    def log_point_cloud(self, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray,
                        title: str, epoch: int = -1) -> None:
        """
        Log a point cloud to the WandB logger.

        Arguments:
        ----------
        X_train : np.ndarray
            The training data.
        X_test : np.ndarray
            The testing data.
        y_train : np.ndarray
            The training labels.
        y_test : np.ndarray
            The testing labels.
        title : str
            The title of the plot.
        epoch : int
            The epoch number.
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
        self.log_images: bool = config.logging.log_images
        self.override_existing: bool = config.logging.override_existing
        self.save_dir: str = config.logging.save_dir
        self.verbose: bool = config.logging.verbose

        self.experiment_dir: str = os.path.join(self.save_dir, self.experiment_name)
        self.epochs_dir: str = os.path.join(self.experiment_dir, "epochs")

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

    def log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Log metrics to the logger.

        Arguments:
        ----------
        metrics : Dict[str, float]
            The metrics to log.
        epoch : int
            The epoch number.
        """
        metrics_df: pd.DataFrame = pd.DataFrame(metrics, index=[epoch])
        metrics_fp: str = os.path.join(self.experiment_dir, "metrics.csv")
        if os.path.exists(metrics_fp):
            metrics_df.to_csv(metrics_fp, mode="a", header=False)
        else:
            metrics_df.to_csv(metrics_fp)

    def log_image(self, image: np.ndarray, caption: str, epoch: int = -1,
                dpi: int = 100) -> None:
        """
        Log an image to the logger.

        Arguments:
        ----------
        image : np.ndarray
            The image to log.
        caption : str
            The caption for the image.
        """
        if not self.log_images:
            return

        plt.imshow(image)
        plt.title(caption)

        if epoch != -1:
            os.makedirs(self.epochs_dir, exist_ok=True)
            epoch_dir: str = os.path.join(self.epochs_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            plt_save_fp: str = os.path.join(epoch_dir, f"{caption}.png")
        else:
            plt_save_fp: str = os.path.join(self.experiment_dir, f"{caption}.png")

        plt.savefig(plt_save_fp, dpi=dpi)
        plt.close()

    def log_point_cloud(self, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray,
                        title: str, epoch: int = -1, ) -> None:
        """
        Log a point cloud to the logger.

        Arguments:
        ----------
        X_train : np.ndarray
            The training data.
        X_test : np.ndarray
            The testing data.
        y_train : np.ndarray
            The training labels.
        y_test : np.ndarray
            The testing labels.
        title : str
            The title of the plot.
        epoch : int
            The epoch number.
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train)
        ax[0].set_title("Training Data")

        ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test)
        ax[1].set_title("Testing Data")

        if epoch != -1:
            os.makedirs(self.epochs_dir, exist_ok=True)
            epoch_dir: str = os.path.join(self.epochs_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            plt_save_fp: str = os.path.join(epoch_dir, f"{title}.png")
        else:
            plt_save_fp: str = os.path.join(self.experiment_dir, f"{title}.png")

        plt.savefig(plt_save_fp)
        plt.close()

    def close(self) -> None:
        """
        close all file handlers
        """
        self.std_out_file.close()
