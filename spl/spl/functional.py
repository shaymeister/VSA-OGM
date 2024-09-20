import numpy as np
import torch
from typing import Union

def bind(vs: Union[torch.tensor, list], device: str) -> torch.tensor:
    """
    Bind all argued vectors into a single representation with circular
    convolution

    Arguments:
    ----------
    1) vs (torch.tensor or list[torch.tensor]): the given vectors that need
        to be bound together; can either be in the form of a tensor with
        shape = (num_vectors, num_vsa_dimensions), or a list of tensors of
        shape = (num_vsa_dimensions)
    2) device (str): where to load, store, and operator on tensors

    Returns:
    --------
    1) vs (torch.tensor): all input vector bound together with
        shape = (num_vsa_dimensions)
    """

    # -----------------------------
    # TODO Add Argument Validation
    # -----------------------------

    if isinstance(vs, list):
        ubvm_shape = (len(vs), vs[0].shape[-1])
        ubvm = torch.zeros(ubvm_shape, device=device)
        for i, value in enumerate(vs):
            ubvm[i, :] = value

        vs = ubvm

    vs = torch.fft.fft(vs, dim=1)
    vs = torch.prod(vs, dim=0)
    vs = torch.fft.ifft(vs).real
    return vs
    
def invert(ssp: torch.tensor) -> torch.tensor:
    """
    Return the pseudo inverse of the argued hypervector

    Arguments:
    ----------
    1) ssp (torch.tensor): a single hypervector based on Plate's
        "Holographic Reduced Representation"; shape = (num_vsa_dimensions)

    Returns:
    --------
    1) ssp (torch.tensor): the pseudo inverse of the argued ssp calculated
        by reversing the indices; shape = (num_vsa_dimensions)
    """

    # -----------------------------
    # TODO Add Argument Validation
    # -----------------------------

    return ssp[-torch.arange(ssp.shape[0])]

def power(ssp: torch.tensor, scalar: float, length_scale: float = 1.0) -> torch.tensor:
    """
    Fractionally bind hypervectors to continuous scalars with exponentiation
    in the Fourier domain

    More information can be found here: https://shorturl.at/jloDO

    Arguments:
    ----------
    1) ssp (torch.tensor): a hypervector based on Plate's "Holographic
        Reduced Representation"; shape = (num_vsa_dimensions)
    2) scalar (float): the continuous value to exponentiate the ssp with
    3) length_scale (float): the width of the kernel

    Returns:
    --------
    x (torch.tensor): the argued ssp fractional bound to the target scaler
    """

    # -----------------------------
    # TODO Add Argument Validation
    # -----------------------------

    x = torch.fft.fft(ssp)
    x = x ** (scalar / length_scale)
    x = torch.fft.ifft(x)
    return x.real

def make_good_unitary(num_dims: int, device: str,
        eps: float = 1e-3) -> torch.tensor:
    """
    create a hyperdimensional vector of unitary length phasers to build the
    quasi-orthogonal algebraic space

    Arguments:
    ----------
    1) num_dims (int): the dimensionality of the vsa
    2) device (str): where to store the tensor
    3) eps (float): the allowable variability in the phase of each phasor

    Returns:
    --------
    1) v (torch.tensor): a one dimensional tensor of unitary phasors
    """

    # -----------------------------
    # TODO Add Argument Validation
    # -----------------------------

    a = torch.rand((num_dims - 1) // 2)
    sign = np.random.choice((-1, +1), len(a))
    
    sign = torch.from_numpy(sign).to(device)
    a = a.to(device)

    phi = sign * torch.pi * (eps + a * (1 - 2 * eps))

    assert torch.all(torch.abs(phi) >= torch.pi * eps)
    assert torch.all(torch.abs(phi) <= torch.pi * (1 - eps))

    fv = torch.zeros(num_dims, dtype=torch.complex64, device=device)
    fv[0] = 1
    fv[1:(num_dims + 1) // 2] = torch.cos(phi) + 1j * torch.sin(phi)
    fv[(num_dims // 2) + 1:] = torch.flip(torch.conj(fv[1:(num_dims + 1) // 2]), dims=[0])
    
    if num_dims % 2 == 0:
        fv[num_dims // 2] = 1

    assert torch.allclose(torch.abs(fv), torch.ones(fv.shape, device=device))
    
    v = torch.fft.ifft(fv)
    v = v.real
    v = v.to(device)
    
    assert torch.allclose(torch.fft.fft(v), fv)
    assert torch.allclose(torch.linalg.norm(v), torch.ones(v.shape, device=device))

    return v