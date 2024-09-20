import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_1d_numpy(
        x: np.ndarray,
        clear_figure: bool = True,
        title: str = "Runtime",
        x_label: str = "Observation",
        y_label: str = "Time (seconds)",
        save_path: str = None,
        show_plot: bool = False) -> None:
    """
    Plot a 1-dimensional numpy array.

    Parameters:
    - x (np.ndarray): The 1-dimensional numpy array to be plotted.
    - clear_figure (bool): Whether to clear the figure before plotting.
        Default is True.
    - title (str): The title of the plot. Default is "Runtime".
    - x_label (str): The label for the x-axis. Default is "Observation".
    - y_label (str): The label for the y-axis. Default is "Time (seconds)".
    - save_path (str): The file path to save the plot. If None, the plot will
        not be saved. Default is None.
    - show_plot (bool): Whether to display the plot. Default is False.

    Returns:
    - None

    """
    if clear_figure:
        plt.clf()

    plt.plot(x, "o")

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show_plot:
        plt.show()
    

def plot_quadrant_boundaries(
        qb_x: torch.tensor,
        qb_y: torch.tensor,
        world_bounds_norm: torch.tensor,
        clear_figure: bool = True,
        quadrant_level: int = -1,
        title_header: str ='Quadrant Boundaries',
        x_label: str = "X (meters)",
        y_label: str = "Y (meters)",
        save_path: str = None,
        show_plot: bool = False) -> None:
    """
    Plot the quadrant boundaries.

    Args:
        qb_x (torch.tensor): The x-coordinates of the quadrant boundaries.
        qb_y (torch.tensor): The y-coordinates of the quadrant boundaries.
        world_bounds_norm (torch.tensor): The normalized world bounds.
        clear_figure (bool, optional): Whether to clear the figure before
            plotting. Defaults to True.
        quadrant_level (int, optional): The level of the quadrant.
            Defaults to -1.
        title_header (str, optional): The title header of the plot.
            Defaults to 'Quadrant Boundaries'.
        x_label (str, optional): The label for the x-axis. Defaults
            to "X (meters)".
        y_label (str, optional): The label for the y-axis. Defaults
            to "Y (meters)".
        save_path (str, optional): The file path to save the plot.
            Defaults to None.
        show_plot (bool, optional): Whether to display the plot.
            Defaults to False.
    """

    if isinstance(qb_x, torch.Tensor):
        qb_x = qb_x.detach().cpu().numpy()

    if isinstance(qb_y, torch.Tensor):
        qb_y = qb_y.detach().cpu().numpy()

    if isinstance(world_bounds_norm, torch.Tensor):
        world_bounds_norm = world_bounds_norm.detach().cpu().numpy()

    if clear_figure:
        plt.clf()

    # Visually display the vertical quadrant boundaries
    for i in range(qb_x.shape[0]):
        plt.plot([qb_x[i], qb_x[i]], [qb_y[0], qb_y[-1]], 'k')
    
    # Visually display the horizontal quadrant boundaries
    for i in range(qb_y.shape[0]):
        plt.plot([qb_x[0], qb_x[-1]], [qb_y[i], qb_y[i]], 'k')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xlim(0, world_bounds_norm[0])
    plt.ylim(0, world_bounds_norm[1])

    if qb_x.shape[0] < 10:
        plt.xticks(qb_x)
    
    if qb_y.shape[0] < 10:
        plt.yticks(qb_y)

    if quadrant_level != -1:
        plt.title(f"{title_header} - Level {quadrant_level}")
    else:
        plt.title(title_header)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show_plot:
        plt.show()


def plot_quadrant_centers(
        qcs: torch.tensor,
        world_bounds_norm: torch.tensor,
        clear_figure: bool = True,
        quadrant_level: int = -1,
        title_header: str ='Quadrant Centers',
        x_label: str = "X (meters)",
        y_label: str = "Y (meters)",
        save_path: str = None,
        show_plot: bool = False) -> None:
    """
    Plots the quadrant centers.

    Args:
        qcs (torch.tensor): The tensor containing the quadrant centers.
        world_bounds_norm (torch.tensor): The tensor containing the normalized
            world bounds.
        clear_figure (bool, optional): Whether to clear the figure before
            plotting. Defaults to True.
        quadrant_level (int, optional): The level of the quadrant.
            Defaults to -1.
        title_header (str, optional): The title header for the plot. Defaults
            to 'Quadrant Centers'.
        x_label (str, optional): The label for the x-axis. Defaults
            to "X (meters)".
        y_label (str, optional): The label for the y-axis. Defaults
            to "Y (meters)".
        save_path (str, optional): The path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults
            to False.

    Returns:
        None
    """

    if isinstance(qcs, torch.Tensor):
        qcs = qcs.detach().cpu().numpy()

    if isinstance(qb_x, torch.Tensor):
        qb_x = qb_x.detach().cpu().numpy()
    
    if isinstance(qb_y, torch.Tensor):
        qb_y = qb_y.detach().cpu().numpy()

    if isinstance(world_bounds_norm, torch.Tensor):
        world_bounds_norm = world_bounds_norm.detach().cpu().numpy()

    if clear_figure:
        plt.clf()
    
    # Visually display the vertical quadrant boundaries
    for i in range(qb_x.shape[0]):
        plt.plot([qb_x[i], qb_x[i]], [qb_y[0], qb_y[-1]], 'k')
    
    # Visually display the horizontal quadrant boundaries
    for i in range(qb_y.shape[0]):
        plt.plot([qb_x[0], qb_x[-1]], [qb_y[i], qb_y[i]], 'k')

    for center in qcs:
        plt.plot(center[0], center[1], 'ro')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xlim(0, world_bounds_norm[0])
    plt.ylim(0, world_bounds_norm[1])

    if qb_x.shape[0] < 10:
        plt.xticks(qb_x)
    
    if qb_y.shape[0] < 10:
        plt.yticks(qb_y)

    if quadrant_level != -1:
        plt.title(f"{title_header} - Level {quadrant_level}")
    else:
        plt.title(title_header)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show_plot:
        plt.show()                           
                   

def plot_2d_heatmap_queried(plane_matrix: torch.tensor,
        axis_bounds: torch.tensor, save_path: str, title: str ='',
        origin: str = "lower", interpolation: str = "none", vmin: int = -1,
        vmax: int = 1, cmap='plasma') -> None:
    """
    Use Matplotlib to display a heatmap based visualization of the similarity
    between the query vector and plane matrix.

    Notes:
    ------
    - both axis of the 2D query, i.e. x and y should be binded together
        into a single vector.
    
    Arguments:
    ----------
    1) plane_matrix (torch.tensor): a three dimensional tensor symbolizing the
        vector representation of each location within the plane;
        shape = (x_length, y_length, num_vsa_dimensions)
    2) axis_bounds (torch.tensor): a one dimensional tensor specifying the
        bounds of each axis; [x_min, x_max, y_min, y_max]
    3) title (str): a string representing the title of the final plot
    4) save_path (str): path to to save the final plot
    5) origin (str): specify the location of the origin in the plot
    6) interpolation (str): how to upscale the final heatmap to different
        resolutions; for more information please see: https://shorturl.at/zBEFV
    7) vmin (int): min value of the color map
    8) vmax (int): max value of the color map
    9) cmap (str): the color scheme of the heat ranges

    Returns:
    --------
    None
    """

    if isinstance(plane_matrix, torch.Tensor):
        plane_matrix = plane_matrix.detach().cpu().numpy()

    plt.clf()
    plt.imshow(
        plane_matrix,
        origin=origin,
        interpolation=interpolation,
        extent=tuple(axis_bounds),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    plt.colorbar()
    plt.title(title)
    plt.savefig(save_path, dpi=300)


def plot_2d_heatmap_unqueried(query: torch.tensor, plane_matrix: torch.tensor,
        axis_bounds: torch.tensor, title: str ='', origin: str = "lower",
        interpolation: str = "none", vmin: int = -1, vmax: int = 1,
        cmap='plasma') -> None:
    """
    Use Matplotlib to display a heatmap based visualization of the similarity
    between the query vector and plane matrix.

    Notes:
    ------
    - both axis of the 2D query, i.e. x and y should be binded together
        into a single vector.
    
    Arguments:
    ----------
    1) query (torch.tensor): a one dimensional vector symbolizing the 2D
        location; shape = (1, num_vsa_dimensions)
    2) plane_matrix (torch.tensor): a three dimensional tensor symbolizing the
        vector representation of each location within the plane;
        shape = (x_length, y_length, num_vsa_dimensions)
    3) axis_bounds (torch.tensor): a one dimensional tensor specifying the
        bounds of each axis; [x_min, x_max, y_min, y_max]
    4) title (str): a string representing the title of the final plot
    5) origin (str): specify the location of the origin in the plot
    6) interpolation (str): how to upscale the final heatmap to different
        resolutions; for more information please see: https://shorturl.at/zBEFV
    7) vmin (int): min value of the color map
    8) vmax (int): max value of the color map
    9) cmap (str): the color scheme of the heat ranges

    Returns:
    --------
    None
    """

    hm = torch.tensordot(query, plane_matrix, dims=([0], [2]))
    hm = hm.cpu().numpy().T

    plt.clf()
    plt.imshow(
        hm,
        origin=origin,
        interpolation=interpolation,
        extent=tuple(axis_bounds),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )
    plt.colorbar()
    plt.title(title)
    plt.show()