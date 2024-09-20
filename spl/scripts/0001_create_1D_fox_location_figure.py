import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

import spl.functional as spf
from spl.generators import SSPGenerator


def parse_args() -> None:
    """
    Parse command line arguments and return a dictionary of parsed arguments.
    
    Returns:
        dict: A dictionary containing the parsed command line arguments.
    """
    description = "1D Figure / Video Generator"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--save_dir",
        default="./scratch",
        dest="save_dir",
        required=False,
        type=str
    )
    parser.add_argument(
        "--seed",
        default=0,
        dest="seed",
        required=False,
        type=int
    )
    
    return vars(parser.parse_args())


def plot_location_sep(memory: torch.tensor, animals: dict,
        x_axis: torch.tensor, x_axis_matrix: torch.tensor, device: str,
        save_path: str, dpi: int = 300) -> None:
    """
    Plots the location probability of a lion based on memory and animal data.

    Args:
        memory (torch.tensor): The memory data.
        animals (dict): A dictionary containing animal data.
        x_axis (torch.tensor): The x-axis data.
        x_axis_matrix (torch.tensor): The x-axis matrix data.
        device (str): The device to use for computation.
        save_path (str): The path to save the plot.
        dpi (int, optional): The resolution of the saved plot. Defaults to 300.

    Returns:
        None
    """

    query = spf.invert(animals["lion"]["symbol"])
    lion_decoded = spf.bind([memory, query], device=device)

    results = torch.sum(lion_decoded * x_axis_matrix, dim=1)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Where is the lion?")

    axs[0].plot(x_axis, results, label="Location Probability")
    axs[0].set_title("Location Probabilities")

    axs[1].plot(x_axis, np.sinc(x_axis), color="orange")
    axs[1].set_title("Ground Truth")
    
    for ax in axs.flat:
        ax.set(xlabel='Location', ylabel='Probability')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(save_path, dpi=dpi)
    plt.clf()


def plot_location_overlay(memory: torch.tensor, animals: dict,
        x_axis: torch.tensor, x_axis_matrix: torch.tensor, device: str,
        save_path: str, dpi: int = 300) -> None:
    """
    Plots the location overlay for a given memory and animals dictionary.

    Args:
        memory (torch.tensor): The memory tensor.
        animals (dict): A dictionary containing animal symbols.
        x_axis (torch.tensor): The x-axis values.
        x_axis_matrix (torch.tensor): The x-axis matrix.
        device (str): The device to use for computation.
        save_path (str): The path to save the plot.
        dpi (int, optional): The resolution of the saved plot. Defaults to 300.

    Returns:
        None
    """

    query = spf.invert(animals["lion"]["symbol"])
    lion_decoded = spf.bind([memory, query], device=device)

    results = torch.sum(lion_decoded * x_axis_matrix, dim=1)

    plt.title("Where is the lion?")
    plt.plot(x_axis, results, label="Location Probability")
    plt.plot(x_axis, np.sinc(x_axis), label="Ground Truth")
    plt.legend()
    plt.savefig(save_path, dpi=dpi)
    plt.clf()


def main(config: dict) -> None:
    """
    Main function for creating a 1D figure or video of lion location.

    Args:
        config (dict): Configuration parameters for the script.

    Returns:
        None
    """

    print(f"----------------------------------")
    print(f"1D Figure / Video Creation Script ")
    print(f"----------------------------------")
    print(f" - Save Directory: {config['save_dir']}")
    print(f" - Seed:           {config['seed']}")

    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    os.makedirs(config["save_dir"], exist_ok=True)

    # ---------------------------------
    # VSA Hyperparameter Configuration
    # ---------------------------------
    axis_resolution: int = 128
    axis_limit: int = 5
    device: str = torch.device("cpu")
    vsa_dimensions: int = 2048

    ssp_gen = SSPGenerator(
        dimensionality=vsa_dimensions,
        device=device
    )

    # -----------------------------------
    # Initialize Spatial Data Structures
    # -----------------------------------
    x_axis = torch.linspace(-axis_limit, axis_limit, axis_resolution, device=device)
    x_axis = x_axis.reshape((-1, 1))

    axis_basis_vectors: torch.tensor = ssp_gen.generate(1)

    x_axis_matrix = torch.zeros((axis_resolution, vsa_dimensions), device=device)
    for i, x in enumerate(x_axis):
        x_axis_matrix[i, :] = spf.power(axis_basis_vectors[0], x)

    # -------------------------------------------
    # Specify Object Locations and Label Vectors
    # ------------------------------------------
    animals: dict = {
        "lion": {
            "locations": torch.tensor([[0]]),
            "symbol": spf.make_good_unitary(vsa_dimensions, device),
        }
    }

    # -----------------------------------------------------
    # Associate each location the location of the lion with 
    # fractionally bound hypervector
    # -----------------------------------------------------
    memory = torch.zeros(vsa_dimensions, device=device)

    for object_key in animals:
        symbol_vector: torch.tensor = animals[object_key]["symbol"]
        locations: torch.tensor = animals[object_key]["locations"]

        # encode each location is cartesian space to hyperdimensional space
        for i in range(locations.shape[0]):
            vs = [
                spf.power(axis_basis_vectors[0], locations[i]),
                symbol_vector
            ]
            memory += spf.bind(vs, device)

    # -----------------------------------------
    # Plot the location of the lion at any time
    # -----------------------------------------
    print(" > Plotting lion location with ground truth - separated")
    path: str = os.path.join(config["save_dir"], "1d_lion_location_no_t_sep.png")
    plot_location_sep(
        memory=memory,
        animals=animals,
        x_axis=x_axis,
        x_axis_matrix=x_axis_matrix,
        device=device,
        save_path=path
    )

    # -----------------------------------------
    # Plot the location of the lion at any time
    # -----------------------------------------
    print(" > Plotting lion location with ground truth - overlaid")
    path: str = os.path.join(config["save_dir"], "1d_lion_location_no_t_overlay.png")
    plot_location_overlay(
        memory=memory,
        animals=animals,
        x_axis=x_axis,
        x_axis_matrix=x_axis_matrix,
        device=device,
        save_path=path
    )


if __name__ == "__main__":
    config: dict = parse_args()
    main(config)
