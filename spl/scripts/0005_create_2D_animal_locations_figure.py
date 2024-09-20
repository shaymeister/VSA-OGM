import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch

import spl.encoders as spe
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


def plot_undecoded_space(memory: torch.tensor, animals: dict,
        axis_bounds: torch.tensor, xy_axis_matrix: torch.tensor,
        save_path: str, origin: str = "lower", interpolation: str = "none",
        vmin: int = -1, vmax: int = 1, cmap='plasma', dpi: int = 300, 
        img_margin: float = 0.75) -> None:
    """
    Plots the undecoded memory space with animal locations.

    Args:
        memory (torch.tensor): The memory tensor.
        animals (dict): A dictionary containing information about the animals.
        axis_bounds (torch.tensor): The bounds of the axis.
        xy_axis_matrix (torch.tensor): The matrix for the xy axis.
        save_path (str): The path to save the plot.
        origin (str, optional): The origin of the plot. Defaults to "lower".
        interpolation (str, optional): The interpolation method.
            Defaults to "none".
        vmin (int, optional): The minimum value for the color map.
            Defaults to -1.
        vmax (int, optional): The maximum value for the color map.
            Defaults to 1.
        cmap (str, optional): The color map. Defaults to 'plasma'.
        dpi (int, optional): The resolution of the saved plot. Defaults to 300.
        img_margin (float, optional): The margin around the animal images.
            Defaults to 0.75.

    Returns:
        None
    """
    hm = torch.tensordot(memory, xy_axis_matrix, dims=([0], [2]))
    hm = hm.cpu().numpy()

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Undecoded Memory Space")

    c = axs[0].imshow(
        hm,
        origin=origin,
        interpolation=interpolation,
        extent=axis_bounds,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    axs[0].set_title("Location Probabilities")

    plt.colorbar(c, ax=axs[0], fraction=0.046, pad=0.04)

    fox_img_path: str = "./assets/animals/lion.png"
    fox_img = Image.open(fox_img_path)
    fox_img = np.array(fox_img)

    chicken_img_path: str = "./assets/animals/hen.png"
    chicken_img = Image.open(chicken_img_path)
    chicken_img = np.array(chicken_img)

    for i in range(animals["fox"]["locations"].shape[0]):
        axs[1].imshow(
            fox_img,
            extent=[
                -animals["fox"]["locations"][i][0] - img_margin,
                -animals["fox"]["locations"][i][0] + img_margin,
                -animals["fox"]["locations"][i][1] - img_margin,
                -animals["fox"]["locations"][i][1] + img_margin,
            ],
            aspect="auto",
            cmap="gray"
        )

    for i in range(animals["chicken"]["locations"].shape[0]):
        axs[1].imshow(
            chicken_img,
            extent=[
                -animals["chicken"]["locations"][i][0] - img_margin,
                -animals["chicken"]["locations"][i][0] + img_margin,
                -animals["chicken"]["locations"][i][1] - img_margin,
                -animals["chicken"]["locations"][i][1] + img_margin,
            ],
            aspect="auto",
            cmap="gray"
        )

    axs[1].set_xlim(-5, 5)
    axs[1].set_ylim(-5, 5)
    axs[1].set_title("Ground Truth")

    asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
    axs[1].set_aspect(asp - 0.1)

    for ax in axs.flat:
        ax.set(xlabel='Location', ylabel='Probability')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=dpi)
    plt.clf()


def plot_fox_location(memory: torch.tensor, animals: dict, device: str,
        axis_bounds: torch.tensor, xy_axis_matrix: torch.tensor,
        save_path: str, origin: str = "lower", interpolation: str = "none",
        vmin: int = -1, vmax: int = 1, cmap='plasma', dpi: int = 300, 
        img_margin: float = 0.75) -> None:
    """
    Plots the location probabilities of foxes and ground truth locations of
    foxes and chickens.

    Args:
        memory (torch.tensor): The memory tensor.
        animals (dict): A dictionary containing information about the animals.
        device (str): The device to be used for computation.
        axis_bounds (torch.tensor): The bounds of the axis.
        xy_axis_matrix (torch.tensor): The xy axis matrix.
        save_path (str): The path to save the figure.
        origin (str, optional): The origin of the plot. Defaults to "lower".
        interpolation (str, optional): The interpolation method.
            Defaults to "none".
        vmin (int, optional): The minimum value for the colorbar.
            Defaults to -1.
        vmax (int, optional): The maximum value for the colorbar.
            Defaults to 1.
        cmap (str, optional): The colormap to be used. Defaults to 'plasma'.
        dpi (int, optional): The resolution of the saved figure.
            Defaults to 300.
        img_margin (float, optional): The margin around the animal images.
            Defaults to 0.75.

    Returns:
        None
    """
    query = spf.invert(animals["fox"]["symbol"])
    fox_decoded = spf.bind([memory, query], device=device)

    hm = torch.tensordot(fox_decoded, xy_axis_matrix, dims=([0], [2]))
    hm = hm.cpu().numpy()

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Where are the foxes?")

    c = axs[0].imshow(
        hm,
        origin=origin,
        interpolation=interpolation,
        extent=axis_bounds,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    axs[0].set_title("Location Probabilities")

    plt.colorbar(c, ax=axs[0], fraction=0.046, pad=0.04)

    fox_img_path: str = "./assets/animals/lion.png"
    fox_img = Image.open(fox_img_path)
    fox_img = np.array(fox_img)

    chicken_img_path: str = "./assets/animals/hen.png"
    chicken_img = Image.open(chicken_img_path)
    chicken_img = np.array(chicken_img)

    for i in range(animals["fox"]["locations"].shape[0]):
        axs[1].imshow(
            fox_img,
            extent=[
                -animals["fox"]["locations"][i][0] - img_margin,
                -animals["fox"]["locations"][i][0] + img_margin,
                -animals["fox"]["locations"][i][1] - img_margin,
                -animals["fox"]["locations"][i][1] + img_margin,
            ],
            aspect="auto",
            cmap="gray"
        )

    for i in range(animals["chicken"]["locations"].shape[0]):
        axs[1].imshow(
            chicken_img,
            extent=[
                -animals["chicken"]["locations"][i][0] - img_margin,
                -animals["chicken"]["locations"][i][0] + img_margin,
                -animals["chicken"]["locations"][i][1] - img_margin,
                -animals["chicken"]["locations"][i][1] + img_margin,
            ],
            aspect="auto",
            cmap="gray"
        )

    axs[1].set_xlim(-5, 5)
    axs[1].set_ylim(-5, 5)
    axs[1].set_title("Ground Truth")

    asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
    axs[1].set_aspect(asp - 0.1)

    for ax in axs.flat:
        ax.set(xlabel='Location', ylabel='Probability')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=dpi)
    plt.clf()


def plot_chicken_location(memory: torch.tensor, animals: dict, device: str,
        axis_bounds: torch.tensor, xy_axis_matrix: torch.tensor,
        save_path: str, origin: str = "lower", interpolation: str = "none",
        vmin: int = -1, vmax: int = 1, cmap='plasma', dpi: int = 300, 
        img_margin: float = 0.75) -> None:
    """
    Plots the location probabilities of chickens and ground truth locations of
    animals.

    Args:
        memory (torch.tensor): The memory tensor.
        animals (dict): A dictionary containing information about the animals.
        device (str): The device to be used for computation.
        axis_bounds (torch.tensor): The bounds of the axis.
        xy_axis_matrix (torch.tensor): The xy axis matrix.
        save_path (str): The path to save the plot.
        origin (str, optional): The origin of the plot. Defaults to "lower".
        interpolation (str, optional): The interpolation method.
            Defaults to "none".
        vmin (int, optional): The minimum value for the colorbar.
            Defaults to -1.
        vmax (int, optional): The maximum value for the colorbar.
            Defaults to 1.
        cmap (str, optional): The colormap to be used. Defaults to 'plasma'.
        dpi (int, optional): The resolution of the saved plot. Defaults to 300.
        img_margin (float, optional): The margin around the animal images.
            Defaults to 0.75.

    Returns:
        None
    """
    query = spf.invert(animals["chicken"]["symbol"])
    fox_decoded = spf.bind([memory, query], device=device)

    hm = torch.tensordot(fox_decoded, xy_axis_matrix, dims=([0], [2]))
    hm = hm.cpu().numpy()

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Where are the chickens?")

    c = axs[0].imshow(
        hm,
        origin=origin,
        interpolation=interpolation,
        extent=axis_bounds,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    axs[0].set_title("Location Probabilities")

    plt.colorbar(c, ax=axs[0], fraction=0.046, pad=0.04)

    fox_img_path: str = "./assets/animals/lion.png"
    fox_img = Image.open(fox_img_path)
    fox_img = np.array(fox_img)

    chicken_img_path: str = "./assets/animals/hen.png"
    chicken_img = Image.open(chicken_img_path)
    chicken_img = np.array(chicken_img)

    for i in range(animals["fox"]["locations"].shape[0]):
        axs[1].imshow(
            fox_img,
            extent=[
                -animals["fox"]["locations"][i][0] - img_margin,
                -animals["fox"]["locations"][i][0] + img_margin,
                -animals["fox"]["locations"][i][1] - img_margin,
                -animals["fox"]["locations"][i][1] + img_margin,
            ],
            aspect="auto",
            cmap="gray"
        )

    for i in range(animals["chicken"]["locations"].shape[0]):
        axs[1].imshow(
            chicken_img,
            extent=[
                -animals["chicken"]["locations"][i][0] - img_margin,
                -animals["chicken"]["locations"][i][0] + img_margin,
                -animals["chicken"]["locations"][i][1] - img_margin,
                -animals["chicken"]["locations"][i][1] + img_margin,
            ],
            aspect="auto",
            cmap="gray"
        )

    axs[1].set_xlim(-5, 5)
    axs[1].set_ylim(-5, 5)
    axs[1].set_title("Ground Truth")

    asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
    axs[1].set_aspect(asp - 0.1)

    for ax in axs.flat:
        ax.set(xlabel='Location', ylabel='Probability')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=dpi)
    plt.clf()


def main(config: dict) -> None:
    """
    Generate a 2D figure and video based on animal locations.

    Args:
        config (dict): Configuration parameters for the script.

    Returns:
        None
    """

    print(f"----------------------------------")
    print(f"2D Figure / Video Creation Script ")
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

    y_axis = torch.linspace(-axis_limit, axis_limit, axis_resolution, device=device)
    y_axis = y_axis.reshape((-1, 1))

    axis_basis_vectors: torch.tensor = ssp_gen.generate(2)

    xy_axis_matrix = torch.zeros((axis_resolution, axis_resolution, vsa_dimensions), device=device)
    for i, x in enumerate(x_axis):
        for j, y in enumerate(y_axis):
            vs = [spf.power(axis_basis_vectors[0], x), spf.power(axis_basis_vectors[1], y)]
            xy_axis_matrix[i, j, :] = spf.bind(vs, device)

    # -------------------------------------------
    # Specify Object Locations and Label Vectors
    # ------------------------------------------
    animals: dict = {
        "fox": {
            "locations": torch.tensor([[-4, -4], [4, 4]]),
            "symbol": spf.make_good_unitary(vsa_dimensions, device)
        },
        "chicken": {
            "locations": torch.tensor([[2, -2]]),
            "symbol": spf.make_good_unitary(vsa_dimensions, device)
        }
    }

    axis_bounds = torch.tensor([-axis_limit, axis_limit, -axis_limit, axis_limit])

    # --------------
    # Create Memory
    # --------------
    memory = torch.zeros(vsa_dimensions)

    for object_key in animals:
        symbol_vector: torch.tensor = animals[object_key]["symbol"]
        locations: torch.tensor = animals[object_key]["locations"]

        # encode each location is cartesian space to hyperdimensional space
        location_vectors = torch.zeros((locations.shape[0], vsa_dimensions))
        for i in range(locations.shape[0]):
            location_vectors[i, :] = spe.encode_cartesian(locations[i], axis_basis_vectors, device=device)

        # bind each location vector with the associated symbol
        for i in range(locations.shape[0]):
            vs = [location_vectors[i, :], symbol_vector]
            location_vectors[i, :] = spf.bind(vs, device)

        for i in range(locations.shape[0]):
            memory += location_vectors[i, :]

    # -----------------------------
    # Plot un-decoded vector space
    # -----------------------------
    print(" > Plotting undecoded vector space")
    path: str = os.path.join(config["save_dir"], "2d_animals_locations_none.png")
    plot_undecoded_space(
        memory=memory,
        animals=animals,
        axis_bounds=axis_bounds,
        xy_axis_matrix=xy_axis_matrix,
        device=device,
        save_path=path
    )

    # -------------------
    # Plot fox locations
    # -------------------
    print(" > Plotting fox locations")
    path: str = os.path.join(config["save_dir"], "2d_animals_locations_fox.png")
    plot_fox_location(
        memory=memory,
        animals=animals,
        axis_bounds=axis_bounds,
        xy_axis_matrix=xy_axis_matrix,
        device=device,
        save_path=path
    )

    # -----------------------
    # Plot chicken locations
    # -----------------------
    print(" > Plotting chicken locations")
    path: str = os.path.join(config["save_dir"], "2d_animals_locations_chicken.png")
    plot_chicken_location(
        memory=memory,
        animals=animals,
        axis_bounds=axis_bounds,
        xy_axis_matrix=xy_axis_matrix,
        device=device,
        save_path=path
    )

if __name__ == "__main__":
    config: dict = parse_args()
    main(config)
