import numpy as np
import os
import pickle as pkl
from sklearn import metrics

from vsa_ogm.plotting import plot_AUC
from spl.mapping import OGM2D_V4


def calculate_AUC(mapper: OGM2D_V4, test_data: dict, log_dir: str,
        threshold_range: list[float] = [-1, 1]) -> None:
    """
    Calculate the Area Under the Curve (AUC) for a given mapper and test data.

    Args:
        mapper (OGM2D_V3): The mapper object used for prediction.
        test_data (dict): The test data containing lidar data and labels.
        log_dir (str): The directory to save the log files.
        threshold_range (list): The range of threshold values to evaluate.

    Raises:
        NotImplementedError: This function is not fully implemented yet.
    """

    assert isinstance(mapper, OGM2D_V4)
    assert isinstance(test_data, dict)
    assert "lidar_data" in test_data.keys()
    assert "occupancy" in test_data.keys()
    assert isinstance(test_data["lidar_data"], list)
    assert isinstance(test_data["occupancy"], list)
    assert len(test_data["lidar_data"]) == len(test_data["occupancy"])
    assert isinstance(log_dir, str)
    assert os.path.isdir(log_dir)

    y_true: list[np.ndarray] = test_data["occupancy"]
    y_true: np.ndarray = np.concatenate(y_true)

    points: list[np.ndarray] = test_data["lidar_data"]
    points: np.ndarray = np.concatenate(points)

    threshold_range_array: np.ndarray = np.arange(threshold_range[0], threshold_range[1], 0.001)

    y_true_list: list[np.ndarray] = [y_true] * threshold_range_array.shape[0]
    y_pred_list: list[np.ndarray] = []

    for threshold in threshold_range_array:
        y_pred: list[np.ndarray] = calculate_y_pred(mapper, points, threshold)
        y_pred_list.append(y_pred)

    tpr_list, fpr_list = calculate_multiple_TP_FP_rates(y_true_list, y_pred_list)

    auc: float = metrics.auc(fpr_list, tpr_list)

    plot_AUC(fpr_list, tpr_list, auc, log_dir)

    tpr_save_path: str = os.path.join(log_dir, "tpr.pkl")
    fpr_save_path: str = os.path.join(log_dir, "fpr.pkl")

    with open(tpr_save_path, "wb") as f:
        pkl.dump(tpr_list, f)

    with open(fpr_save_path, "wb") as f:
        pkl.dump(fpr_list, f)


def calculate_y_pred(mapper: OGM2D_V4, test_data: np.ndarray,
        threshold: float) -> list[np.ndarray]:
    """
    Calculate the predicted occupancy values based on a given mapper, testing
    data, and threshold.

    Args:
        mapper (OGM2D_V3): The mapper object used for prediction.
        test_data (np.ndarray): The test data containing lidar data and classes.
        threshold (float): The threshold value for classifying occupancy.

    Returns:
        list[np.ndarray]: The predicted occupancy values with the threshold.
    """
    assert isinstance(mapper, OGM2D_V4)
    assert isinstance(test_data, np.ndarray)
    assert isinstance(threshold, float)

    # filter out all data that is outside the world bounds
    test_data = test_data[test_data[:,0] >= mapper.world_bounds[0], :]
    test_data = test_data[test_data[:,0] <= mapper.world_bounds[1], :]
    test_data = test_data[test_data[:,1] >= mapper.world_bounds[2], :]
    test_data = test_data[test_data[:,1] <= mapper.world_bounds[3], :]

    y_pred: np.ndarray = np.zeros(test_data.shape[0])

    point_thetas: np.ndarray = mapper.query_point_thetas(test_data)

    y_pred[point_thetas >= threshold] = 1
    y_pred[point_thetas < threshold] = 0

    return y_pred


def calculate_multiple_TP_FP_rates(y_true: list[np.ndarray],
        y_pred: list[np.ndarray]) -> tuple:
    """
    Calculate the true positive rate (TPR) and false positive rate (FPR) for
    multiple sets of true labels and predicted labels.

    Args:
        y_true (list[np.ndarray]): A list containing the true labels.
        y_pred (list[np.ndarray]): A list containing the predicted labels.

    Returns:
        tuple: A tuple containing two lists: the true positive rates (TPRs) and
            false positive rates (FPRs) for each set of labels.
    """
    tpr_list: list[float] = []
    fpr_list: list[float] = []

    for i in range(len(y_true)):
        tpr, fpr = calculate_TP_FP_rate(y_true[i], y_pred[i])
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return tpr_list, fpr_list


def calculate_TP_FP_rate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculates the true positive rate and false positive rate for a given set of labels and predictions.

    Args:
        y_true (np.ndarray): A numpy array containing the true labels.
        y_pred (np.ndarray): A numpy array containing the predicted labels.

    Returns:
        (float, float): A tuple containing the true positive rate and false positive rate.
    """

    # -----------------------------
    # TODO Add Argument Validation
    # -----------------------------

    # calculate the number of true positives, false positives, and false negatives
    TP: int = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    TN: int = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    FP: int = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    FN: int = np.sum(np.logical_and(y_true == 1, y_pred == 0))

    # calculate the true positive rate and false positive rate
    TPR: float = TP / (TP + FN)
    FPR: float = FP / (FP + TN)

    return TPR, FPR