import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_args() -> None:
    """
    Parse command line arguments and return a dictionary of parsed arguments.

    Returns:
        dict: A dictionary containing the parsed arguments.
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


def main(config: dict) -> None:
    """
    Generate a 2D lidar spinning animation and save it as a GIF.

    Args:
        config (dict): A dictionary containing configuration parameters.
            - save_dir (str): The directory to save the generated GIF.
            - seed (int): The seed for random number generation.

    Returns:
        None
    """

    print(f"----------------------------------")
    print(f"1D Figure / Video Creation Script ")
    print(f"----------------------------------")
    print(f" - Save Directory: {config['save_dir']}")
    print(f" - Seed:           {config['seed']}")

    os.makedirs(config["save_dir"], exist_ok=True)

    num_points: int = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    lidar_radius: float = 50
    world_radius: float = lidar_radius + 3
    slice: int = 10

    lidar_x: np.ndarray = lidar_radius * np.cos(theta)
    lidar_y: np.ndarray = lidar_radius * np.sin(theta)

    figure_artists = []
    fig, ax = plt.subplots(1, 1)

    i = 0
    while i + slice < num_points:
        i += 1

        a = ax.plot(lidar_x[i:i+slice], lidar_y[i:i+slice], color="blue")
        ax.set_xlim(-world_radius, world_radius)
        ax.set_ylim(-world_radius, world_radius)

        figure_artists.append(a)
        
    ani = animation.ArtistAnimation(
        fig=fig,
        artists=figure_artists,
        interval=50,
        blit=True,
        repeat_delay=1000
    )

    video_writer = animation.FFMpegWriter(fps=15)
    save_path: str = os.path.join(config["save_dir"], "2d_lidar_spinning.gif")
    ani.save(save_path, video_writer)


if __name__ == "__main__":
    config: dict = parse_args()
    main(config)