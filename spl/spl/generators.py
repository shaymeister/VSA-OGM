import torch

from .functional import make_good_unitary

class SSPGenerator:
    """
    A Utility class to generate arbitrary numbers of hyper-vectors with the
    same shape so they can be binded and bundled together
    """
    def __init__(self, dimensionality: int, device: str, length_scale: float = 1) -> None:
        """
        Init SSP Generator

        Arguments:
        ----------
        1) dimensionality (int): the number of dimensions contained within
            each hypervector
        2) device (str): a string representing the device to load, store,
            and operate
        3) length_scale (float): adjust the width of the kernel

        Returns:
        --------
        None
        """
        self.dimensionality: int = dimensionality
        self.device: str = device
        self.length_scale: float = length_scale

    def generate(self, n: int) -> torch.tensor:
        """
        Randomly create a series of n hypervectors

        Arguments:
        ----------
        1) n (int): the number of vectors to generate

        Returns:
        --------
        1) ssp_matrix (torch.tensor): a matrix of random hypervectors of
            shape [n, self.dimensionality]
        """
        ssp_matrix = torch.zeros((n, self.dimensionality), device=self.device)

        for i in range(n):
            ssp_matrix[i, :] = make_good_unitary(
                num_dims=self.dimensionality,
                device=self.device,
                length_scale=self.length_scale
            )

        return ssp_matrix
