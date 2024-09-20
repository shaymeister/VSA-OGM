import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch

import spl.functional as spf
from spl.generators import SSPGenerator


def parse_args() -> None:
    """
    Parse command line arguments and return them as a dictionary.

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


def plot_ground_truth_any_t(memory_time: torch.tensor, animals: dict,
        x_axis: torch.tensor, x_axis_matrix: torch.tensor, device: str,
        save_path: str, time_basis_vector: torch.tensor, dpi: int = 300,
        img_margin: float = 0.75) -> None:
    """
    Plot where the lion has been at all times (t = ?)

    Args:
        memory_time (torch.tensor): The memory time tensor.
        animals (dict): A dictionary containing information about the animals.
        x_axis (torch.tensor): The x-axis tensor.
        x_axis_matrix (torch.tensor): The x-axis matrix tensor.
        device (str): The device to be used.
        save_path (str): The path to save the plot.
        time_basis_vector (torch.tensor): The time basis vector tensor.
        dpi (int, optional): The DPI (dots per inch) for the saved plot.
            Defaults to 300.
        img_margin (float, optional): The margin for the lion image.
            Defaults to 0.75.

    Returns:
        None: This function does not return anything.
    """

    vs = [animals["lion"]["symbol"], spf.power(time_basis_vector, 0)]
    query = spf.bind(vs, device)
    query = spf.invert(query)

    vs = [memory_time, query]
    lion_decoded_t0 = spf.bind(vs, device)

    vs = [animals["lion"]["symbol"], spf.power(time_basis_vector, 1)]
    query = spf.bind(vs, device)
    query = spf.invert(query)

    vs = [memory_time, query]
    lion_decoded_t1 = spf.bind(vs, device)

    lion_decoded = lion_decoded_t0 + lion_decoded_t1

    results = torch.sum(lion_decoded * x_axis_matrix, dim=1)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Where has the lion been (t = ?)")

    axs[0].plot(x_axis, results, label="Location Probability")
    axs[0].set_title("Location Probabilities")

    lion_img_path: str = "./assets/animals/lion.png"
    lion_img = Image.open(lion_img_path)
    lion_img = np.array(lion_img)

    for i in range(animals["lion"]["locations"].shape[0]):
        axs[1].imshow(
            lion_img,
            extent=[
                animals["lion"]["locations"][i][0] - img_margin,
                animals["lion"]["locations"][i][0] + img_margin,
                0.45, 0.55],
            aspect="auto",
            cmap="gray"
        )

    axs[1].set_xlim(-5, 5)
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Ground Truth")
    
    for ax in axs.flat:
        ax.set(xlabel='Location', ylabel='Probability')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(save_path, dpi=dpi)


def plot_ground_truth_each_t(memory_time: torch.tensor, animals: dict,
        x_axis: torch.tensor, x_axis_matrix: torch.tensor, device: str,
        save_path: str, time_basis_vector: torch.tensor, dpi: int = 300,
        img_margin: float = 0.75) -> None:
    """
    Plot where the lion was at each time step

    Args:
        memory_time (torch.tensor): The memory time tensor.
        animals (dict): A dictionary containing information about the animals.
        x_axis (torch.tensor): The x-axis tensor.
        x_axis_matrix (torch.tensor): The x-axis matrix tensor.
        device (str): The device to use for computation.
        save_path (str): The path to save the generated plots.
        time_basis_vector (torch.tensor): The time basis vector tensor.
        dpi (int, optional): The DPI (dots per inch) for the saved plots.
            Defaults to 300.
        img_margin (float, optional): The margin around the lion image.
            Defaults to 0.75.

    Returns:
        None
    """

    for i in range(animals["lion"]["locations"].shape[0]):
        vs = [spf.power(time_basis_vector, i), animals["lion"]["symbol"]]
        query = spf.bind(vs, device)
        query = spf.invert(query)

        vs = [memory_time, query]
        lion_decoded = spf.bind(vs, device)

        results = torch.sum(lion_decoded * x_axis_matrix, dim=1)

        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f"Where has the lion been (t = {i})")

        axs[0].plot(x_axis, results, label="Location Probability")
        axs[0].set_title("Location Probabilities")

        lion_img_path: str = "./assets/animals/lion.png"
        lion_img = Image.open(lion_img_path)
        lion_img = np.array(lion_img)

        axs[1].imshow(
            lion_img,
            extent=[
                animals["lion"]["locations"][i][0] - img_margin,
                animals["lion"]["locations"][i][0] + img_margin,
                0.45, 0.55],
            aspect="auto",
            cmap="gray"
        )

        axs[1].set_xlim(-5, 5)
        axs[1].set_ylim(0, 1)
        axs[1].set_title("Ground Truth")
        
        for ax in axs.flat:
            ax.set(xlabel='Location', ylabel='Probability')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.savefig(save_path + f"{i}.png", dpi=dpi)

def main(config: dict) -> None:
    """
    Main function for creating a 1D figure or video.

    Args:
        config (dict): Configuration parameters for the figure or video creation.

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
    time_basis_vector: torch.tensor = ssp_gen.generate(1)

    x_axis_matrix = torch.zeros((axis_resolution, vsa_dimensions), device=device)
    for i, x in enumerate(x_axis):
        x_axis_matrix[i, :] = spf.power(axis_basis_vectors[0], x)

    animals: dict = {
        "lion": {
            "locations": torch.tensor([[-4], [4]]),
            "symbol": spf.make_good_unitary(vsa_dimensions, device),
        }
    }

    # ----------------------------------------------------------
    # Associate each location of the lion with a given time step
    # ----------------------------------------------------------
    memory_time = torch.zeros(vsa_dimensions, device=device)

    for object_key in animals:
        symbol_vector: torch.tensor = animals[object_key]["symbol"]
        locations: torch.tensor = animals[object_key]["locations"]

        # encode each location is cartesian space to hyperdimensional space
        for i in range(locations.shape[0]):
            vs = [
                spf.bind([spf.power(axis_basis_vectors[0], locations[i])], device=device),
                spf.power(time_basis_vector, i),
                symbol_vector
            ]
            memory_time += spf.bind(vs, device)

    # -----------------------------------------
    # Plot the location of the lion at any time
    # -----------------------------------------
    print(" > Plotting lion locations at any time step")
    path: str = os.path.join(config["save_dir"], "1d_lion_locations_t_any.png")
    plot_ground_truth_any_t(
        memory_time=memory_time,
        time_basis_vector=time_basis_vector,
        animals=animals,
        x_axis=x_axis,
        x_axis_matrix=x_axis_matrix,
        device=device,
        save_path=path
    )

    # -----------------------------------------------
    # Plot the location of the lion at each time step
    # -----------------------------------------------
    print(" > Plotting lion location at each time step")
    path: str = os.path.join(config["save_dir"], "1d_lion_locations_t")
    plot_ground_truth_each_t(
        memory_time=memory_time,
        animals=animals,
        x_axis=x_axis,
        x_axis_matrix=x_axis_matrix,
        device=device,
        save_path=path,
        time_basis_vector=time_basis_vector
    )


if __name__ == "__main__":
    config: dict = parse_args()
    main(config)
