import matplotlib.pyplot as plt
import numpy as np
import os

def plot_AUC(fpr: np.ndarray, tpr: np.ndarray, auc: float, log_dir: str) -> None:
    """
    Plot the Area Under the Curve (AUC) for a given mapper and test data.

    Args:
        fpr (np.ndarray): The false positive rates for each threshold.
        tpr (np.ndarray): The true positive rates for each threshold.
        auc (float): The area under the curve.
        log_dir (str): The directory to save the log files.

    Returns:
        None
    """

    assert isinstance(fpr, list)
    assert isinstance(tpr, list)
    assert isinstance(auc, float)
    assert isinstance(log_dir, str)
    assert os.path.isdir(log_dir)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC: {auc}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("Area Under the Curve (AUC)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(log_dir, "AUC.png"))
    plt.close()