import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics.pairwise import rbf_kernel
import torch

from spl.mapping import LidarOccupancyGridMapper2D


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

def sigmoid(x):
    """
    Compute the sigmoid function of the input x.

    Parameters:
        x (float): The input value.

    Returns:
        float: The sigmoid of x.
    """
    return 1. / (1 + np.exp(-x))
    
def calcPosterior(Phi, y, xi, mu0, sig0):
    """
    Calculates the posterior mean and variance given the input parameters.

    Parameters:
        Phi (numpy.ndarray): The design matrix.
        y (numpy.ndarray): The observed data.
        xi (float): The hyperparameter.
        mu0 (float): The prior mean.
        sig0 (float): The prior variance.

    Returns:
        tuple: A tuple containing the posterior mean and variance.
    """
    logit_inv = sigmoid(xi)
    lam = 0.5 / xi * (logit_inv - 0.5)

    sig = 1. /(1./sig0 + 2*np.sum( (Phi.T**2)*lam, axis=1)) # note the numerical trick for the dot product

    mu = sig*(mu0/sig0 + np.dot(Phi.T, y - 0.5).ravel())

    return mu, sig


def calculate_bhm(lidar_x: torch.tensor, lidar_y: torch.tensor) -> tuple:
    """
    Calculate the Bayesian Occupancy Grid Map (BHM) using simulated lidar data.

    Args:
        lidar_x (torch.tensor): The x-coordinates of the lidar measurements.
        lidar_y (torch.tensor): The y-coordinates of the lidar measurements.

    Returns:
        tuple: A tuple containing the x-coordinates of the grid points and the
            mean occupancy probabilities.
    """

    # Step 0 - data
    X3 = np.zeros((lidar_x.shape[0], 2))
    X3[:, 0] = lidar_x
    X3[:, 1] = lidar_y

    y3 = np.ones((lidar_x.shape[0]))

    # Step 1 - define hinde points
    xx, yy = np.meshgrid(np.linspace(-60, 60, 8), np.linspace(-60, 60, 8))
    grid = np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)))

    # Step 2 - compute features
    gamma = 0.7
    Phi = rbf_kernel(X3, grid, gamma=gamma)

    print(Phi.shape)

    # Step 3 - estimate the parameters
    # Let's define the prior
    N, D = Phi.shape[0], Phi.shape[1]
    epsilon = np.ones(N)
    mu = np.zeros(D)
    sig = 10000*np.ones(D)

    for _ in range(3):
        # E-step
        mu, sig = calcPosterior(Phi, y3, epsilon, mu, sig)

        # M-step
        epsilon = np.sqrt(np.sum((Phi**2)*sig, axis=1) + (Phi.dot(mu.reshape(-1, 1))**2).ravel())

    # Step 4 - predict
    qxx, qyy = np.meshgrid(np.linspace(-60, 60, 120), np.linspace(-60, 60, 120))
    qX = np.hstack((qxx.ravel().reshape(-1,1), qyy.ravel().reshape(-1,1)))
    qPhi = rbf_kernel(qX, grid, gamma=gamma)
    qw = np.random.multivariate_normal(mu, np.diag(sig), 1000)
    occ = sigmoid(qw.dot(qPhi.T))
    occMean = np.mean(occ, axis=0)
    # occStdev = np.std(occ, axis=0)

    return qX, occMean



class Lidar2MemoryComparisonFigure:
    """
    A class representing a figure that compares Lidar occupancy grid mapping
    in VSA (Vector Symbolic Architecture) versus BHM (Bayesian Hilbert Map).

    Attributes:
        num_points (int): The number of Lidar points.
        world_radius (float): The radius of the world.
        slice (int): The slice of Lidar points to consider.
        lidar_x (np.ndarray): The x-coordinates of the Lidar points.
        lidar_y (np.ndarray): The y-coordinates of the Lidar points.
        num_cols (int): The number of columns in the figure.
        fig (matplotlib.figure.Figure): The figure object.
        axs (numpy.ndarray): The array of subplot axes.
        axis_titles (list): The titles of the subplots.
        vsa_mapper (LidarOccupancyGridMapper2D): The VSA mapper object.

    Methods:
        __init__(): Initializes the Lidar2MemoryComparisonFigure object.
        __call__(time: int) -> tuple: Updates and returns the Lidar, VSA, and
            BHM plots.
    """
    def __init__(self) -> None:
        """
        Initialize the LidarOccupancyGridMapping class.

        This method sets up the necessary variables and parameters for the class.

        Parameters:
            None

        Returns:
            None
        """

        self.num_points: int = 50
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        lidar_radius: float = 50
        self.world_radius: float = lidar_radius + 3
        self.slice: int = 1

        self.lidar_x: np.ndarray = lidar_radius * np.cos(theta)
        self.lidar_y: np.ndarray = lidar_radius * np.sin(theta)

        x_range = np.linspace(-35, 35, 20)

        self.lidar_x = np.concatenate((self.lidar_x, x_range))
        self.lidar_y = np.concatenate((self.lidar_y, x_range))

        self.lidar_x = np.concatenate((self.lidar_x, x_range))
        self.lidar_y = np.concatenate((self.lidar_y, -x_range))

        self.num_cols: int = 3
        self.fig, self.axs = plt.subplots(1, self.num_cols)
        self.fig.set_figwidth(15)
        self.fig.suptitle("Lidar Occupancy Grid Mapping in VSA versus BHM")
        self.axis_titles = [
            "Lidar Hit Points",
            "VSA Map",
            "Bayesian Hilbert Map"
        ]

        self.vsa_mapper = LidarOccupancyGridMapper2D()

    def __call__(self, time: int) -> tuple:
        """
        Perform the necessary computations and generate visualizations for a
        given time.

        Args:
            time (int): The time index.

        Returns:
            tuple: A tuple containing the lidar plot, vsa_hm plot, and bhm plot.
        """

        assert time + self.slice <= self.lidar_x.shape[0]

        # reset all subplots and give them the same formatting
        for i in range(self.num_cols):
            self.axs[i].cla()
            self.axs[i].set_xlim(-self.world_radius, self.world_radius)
            self.axs[i].set_ylim(-self.world_radius, self.world_radius)
            self.axs[i].set_title(self.axis_titles[i])

        # -----------------------------
        # Plot the Lidar Visualization
        # -----------------------------
        lidar_plot = self.axs[0].plot(
            self.lidar_x[time:time+self.slice+1],
            self.lidar_y[time:time+self.slice+1],
            color="blue"
        )

        lidar_matrix = torch.zeros((self.slice, 2))
        lidar_matrix[:, 1] = torch.from_numpy(self.lidar_x[time:time+self.slice])
        lidar_matrix[:, 0] = torch.from_numpy(self.lidar_y[time:time+self.slice])
        self.vsa_mapper.process_lidar(lidar_matrix)
        query, xy_axis_matrix = self.vsa_mapper.decode_occupied()

        hm = torch.tensordot(query, xy_axis_matrix, dims=([0], [2]))
        hm = hm.cpu().numpy()

        vsa_hm_plot = self.axs[1].imshow(
            hm,
            origin="lower",
            interpolation="none",
            extent=tuple(self.vsa_mapper.axis_bounds),
            vmin=-1,
            vmax=1,
            cmap="plasma"
        )

        qX, occMean = calculate_bhm(
            lidar_x=self.lidar_x[:time+self.slice],
            lidar_y=self.lidar_y[:time+self.slice]
        )

        bhm_plot = self.axs[2].scatter(
            qX[:,0],
            qX[:,1],
            c=occMean,
            s=4,
            cmap='plasma',
            vmin=-1,
            vmax=1
        )

        if time == 1:
            # self.fig.colorbar(vsa_hm_plot, ax=self.axs[1])
            self.fig.colorbar(bhm_plot, ax=self.axs[2])

        return lidar_plot, vsa_hm_plot, bhm_plot


def main(config: dict) -> None:
    """
    Main function for creating a 1D Figure / Video.

    Args:
        config (dict): A dictionary containing configuration parameters.

    Returns:
        None
    """

    print(f"----------------------------------")
    print(f"2D Figure / Video Creation Script ")
    print(f"----------------------------------")
    print(f" - Save Directory: {config['save_dir']}")
    print(f" - Seed:           {config['seed']}")

    os.makedirs(config["save_dir"], exist_ok=True)

    plotting_func = Lidar2MemoryComparisonFigure()

    ani = animation.FuncAnimation(
        fig=plotting_func.fig,
        func=plotting_func,
        interval=50,
        frames=89,
        blit=False, # blitting cannot be used with figure artists
    )

    video_writer = animation.FFMpegWriter(fps=15)
    save_path: str = os.path.join(config["save_dir"], "2d_lidar_spinning_vsa_bhm_ls2_512.gif")
    ani.save(save_path, video_writer)


if __name__ == "__main__":
    config: dict = parse_args()
    main(config)