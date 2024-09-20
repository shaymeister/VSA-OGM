import numpy as np


def train_test_split(data: dict, test_split: float) -> tuple:
    """
    Split the data into training and testing sets.

    Args:
        data (dict): A dictionary containing the lidar data and occupancy data.
        test_split (float): The percentage of the data used for testing.

    Returns:
        tuple: A tuple containing the training lidar data, training occupancy
            data, testing lidar data, and testing occupancy data.
    """

    assert isinstance(data, dict)
    assert "lidar_data" in data.keys()
    assert "occupancy" in data.keys()
    assert isinstance(data["lidar_data"], np.ndarray)
    assert isinstance(data["occupancy"], np.ndarray)
    assert data["lidar_data"].shape[0] == data["occupancy"].shape[0]
    assert isinstance(test_split, float)
    assert test_split > 0.0 and test_split < 1.0

    data_indices: np.ndarray = np.arange(data["lidar_data"].shape[0])
    np.random.shuffle(data_indices)

    test_length: int = int(test_split * data_indices.shape[0])
    
    test_indices: np.ndarray = data_indices[:test_length]
    train_indices: np.ndarray = data_indices[test_length:]

    test_lidar: np.ndarray = data["lidar_data"][test_indices]
    test_occupancy: np.ndarray = data["occupancy"][test_indices]

    train_lidar: np.ndarray = data["lidar_data"][train_indices]
    train_occupancy: np.ndarray = data["occupancy"][train_indices]

    return train_lidar, train_occupancy, test_lidar, test_occupancy
