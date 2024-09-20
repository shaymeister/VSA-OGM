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


def plot_undecoded_space(memory: torch.tensor, animals: dict,
        x_axis: torch.tensor, x_axis_matrix: torch.tensor, device: str,
        save_path: str, dpi: int = 300, img_margin: float = 0.75) -> None:
    """
    Plots the undecoded vector space for multiple animals.

    Args:
        memory (torch.tensor): The memory tensor.
        animals (dict): A dictionary containing information about the animals.
        x_axis (torch.tensor): The x-axis values.
        x_axis_matrix (torch.tensor): The x-axis matrix.
        device (str): The device to use.
        save_path (str): The path to save the plot.
        dpi (int, optional): The DPI (dots per inch) for the saved plot.
            Defaults to 300.
        img_margin (float, optional): The margin around the animal images.
            Defaults to 0.75.

    Returns:
        None
    """

    results = torch.sum(memory * x_axis_matrix, dim=1)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Undecoded Vector Space")

    axs[0].plot(x_axis, results, label="Location Probability")
    axs[0].set_title("Location Probabilities")

    lion_img_path: str = "./assets/animals/lion.png"
    lion_img = Image.open(lion_img_path)
    lion_img = np.array(lion_img)

    chicken_img_path: str = "./assets/animals/hen.png"
    chicken_img = Image.open(chicken_img_path)
    chicken_img = np.array(chicken_img)

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

    for i in range(animals["chicken"]["locations"].shape[0]):
        axs[1].imshow(
            chicken_img,
            extent=[
                animals["chicken"]["locations"][i][0] - img_margin,
                animals["chicken"]["locations"][i][0] + img_margin,
                0.45, 0.55],
            aspect="auto",
            cmap="gray"
        )

    axs[0].set_ylim(-0.1, 1)
    axs[1].set_xlim(-5, 5)
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Ground Truth")        
    for ax in axs.flat:
        ax.set(xlabel='Location', ylabel='Probability')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig(save_path, dpi=dpi)
    plt.clf()


def plot_lion_location(memory: torch.tensor, animals: dict,
        x_axis: torch.tensor, x_axis_matrix: torch.tensor, device: str,
        save_path: str, dpi: int = 300, img_margin: float = 0.75) -> None:
    """
    Plot the location of lions and chickens based on their probabilities.

    Args:
        memory (torch.tensor): The memory tensor.
        animals (dict): A dictionary containing information about the animals.
        x_axis (torch.tensor): The x-axis values.
        x_axis_matrix (torch.tensor): The x-axis matrix.
        device (str): The device to use for computation.
        save_path (str): The path to save the figure.
        dpi (int, optional): The DPI (dots per inch) for the saved figure.
            Defaults to 300.
        img_margin (float, optional): The margin around the animal images.
            Defaults to 0.75.

    Returns:
        None
    """

    query = spf.invert(animals["lion"]["symbol"])
    lion_decoded = spf.bind([memory, query], device=device)

    results = torch.sum(lion_decoded * x_axis_matrix, dim=1)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Where are the lions?")

    axs[0].plot(x_axis, results, label="Location Probability")
    axs[0].set_title("Location Probabilities")

    lion_img_path: str = "./assets/animals/lion.png"
    lion_img = Image.open(lion_img_path)
    lion_img = np.array(lion_img)

    chicken_img_path: str = "./assets/animals/hen.png"
    chicken_img = Image.open(chicken_img_path)
    chicken_img = np.array(chicken_img)

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

    for i in range(animals["chicken"]["locations"].shape[0]):
        axs[1].imshow(
            chicken_img,
            extent=[
                animals["chicken"]["locations"][i][0] - img_margin,
                animals["chicken"]["locations"][i][0] + img_margin,
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
    plt.clf()


def plot_chicken_location(memory: torch.tensor, animals: dict,
        x_axis: torch.tensor, x_axis_matrix: torch.tensor, device: str,
        save_path: str, dpi: int = 300, img_margin: float = 0.75) -> None:
    """
    Plots the location probabilities of chickens and ground truth locations.

    Args:
        memory (torch.tensor): The memory tensor.
        animals (dict): A dictionary containing information about the animals.
        x_axis (torch.tensor): The x-axis values.
        x_axis_matrix (torch.tensor): The x-axis matrix.
        device (str): The device to use for computation.
        save_path (str): The path to save the plot.
        dpi (int, optional): The DPI (dots per inch) for the saved plot.
            Defaults to 300.
        img_margin (float, optional): The margin around the animal images.
            Defaults to 0.75.

    Returns:
        None
    """

    query = spf.invert(animals["chicken"]["symbol"])
    lion_decoded = spf.bind([memory, query], device=device)

    results = torch.sum(lion_decoded * x_axis_matrix, dim=1)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Where are the chickens?")

    axs[0].plot(x_axis, results, label="Location Probability")
    axs[0].set_title("Location Probabilities")

    lion_img_path: str = "./assets/animals/lion.png"
    lion_img = Image.open(lion_img_path)
    lion_img = np.array(lion_img)

    chicken_img_path: str = "./assets/animals/hen.png"
    chicken_img = Image.open(chicken_img_path)
    chicken_img = np.array(chicken_img)

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

    for i in range(animals["chicken"]["locations"].shape[0]):
        axs[1].imshow(
            chicken_img,
            extent=[
                animals["chicken"]["locations"][i][0] - img_margin,
                animals["chicken"]["locations"][i][0] + img_margin,
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
    plt.clf()


def main(config: dict) -> None:
    """
    Main function for creating a 1D figure or video of multiple animal locations.

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
            "locations": torch.tensor([[-4], [4]]),
            "symbol": spf.make_good_unitary(vsa_dimensions, device)
        },
        "chicken": {
            "locations": torch.tensor([[0]]),
            "symbol": spf.make_good_unitary(vsa_dimensions, device)
        }
    }

    # ----------------------------------------------------------
    # Associate each location of the lion with a given time step
    # ----------------------------------------------------------
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

    # -----------------------------
    # Plot un-decoded vector space
    # -----------------------------
    print(" > Plotting undecoded vector space")
    path: str = os.path.join(config["save_dir"], "1d_animals_locations_none.png")
    plot_undecoded_space(
        memory=memory,
        animals=animals,
        x_axis=x_axis,
        x_axis_matrix=x_axis_matrix,
        device=device,
        save_path=path
    )

    # -------------------
    # Plot lion locations
    # -------------------
    print(" > Plotting lion locations")
    path: str = os.path.join(config["save_dir"], "1d_animals_locations_lion.png")
    plot_lion_location(
        memory=memory,
        animals=animals,
        x_axis=x_axis,
        x_axis_matrix=x_axis_matrix,
        device=device,
        save_path=path
    )

    # -----------------------
    # Plot chicken locations
    # -----------------------
    print(" > Plotting chicken locations")
    path: str = os.path.join(config["save_dir"], "1d_animals_locations_chicken.png")
    plot_chicken_location(
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
