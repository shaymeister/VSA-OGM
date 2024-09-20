import torch

from .functional import bind, power

def encode_cartesian(axis_values: torch.tensor, 
        axis_basis_vectors: torch.tensor, device: str,
        length_scale: float = 1.0) -> torch.tensor:
    """
    Encode a series of discrete points along multiple dimensions of
    cartesian space with fractional binding.

    Arguments:
    ----------
    1) axis_values (torch.tensor): a tensor containing a series of discrete
        points along each axis of a cartesian plane;
        shape = (num_axes, num_points)
    2) axis_basis_vectors (torch.tensor): a tensor containing the basis vectors
        for each axis that will be converted to continuous values with
        fractional binding; shape = (num_axes, num_vsa_dimensions)
    3) device (str): a string specifying where to load, store, and operate
        in high dimensional space
    4) length_scale (float): the width of the kernel

    Returns:
    --------
    1) v (torch.tensor): the discrete points along each axis fractionally
        bound into hyperdimensional space; shape = (num_axes, num_vsa_dims)
    """

    assert axis_values.shape[0] == axis_basis_vectors.shape[0]

    ubvm = torch.zeros_like(axis_basis_vectors, device=device)

    # exponentiate each axis basis vector with the desired value
    for i in range(axis_values.shape[0]):
        ubvm[i, :] = power(
            axis_basis_vectors[i, :],
            axis_values[i],
        length_scale)

    v = bind(ubvm, device)
    return v