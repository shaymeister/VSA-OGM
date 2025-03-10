import numpy as np
from omegaconf import DictConfig
import time
import torch
from typing import List

from .base_sa_mapper import BaseSingleAgentMapper
from ...logging import BaseLogger


class SA_BHM_DIAG(BaseSingleAgentMapper):
    """
    TODO Finish Documentation
    """
    # class variables and objects
    num_observations: int = 0
    calc_loss: bool = False
    grid: np.ndarray = None

    scan_no: int = 0
    intercept_: list = [0]
    coef_: list = [0]
    sigma_: list = [0]

    def __init__(self, config: DictConfig, loggers: List[BaseLogger],
                 print_header: str = ("(SA BHM DIAG")) -> None:
        """
        Initialize the VSA_OGM object.

        Args:
            config (DictConfig): The configuration object containing the
                mapping information
            loggers (List[BaseLogger]): A list of loggers to log information.
            print_header (str): The header to print when logging information

        Returns:
            None
        """
        super(SA_BHM_DIAG, self).__init__(config, loggers, print_header)

        # -----------------------------------------------
        # extract parameters from the config
        # -----------------------------------------------
        self.axis_resolution: int = config.mapping.axis_resolution
        self.device: str = config.mapping.device
        self.seed: int = config.mapping.seed
        self.world_bounds: List[int] = config.data.world_bounds
        self.verbose: bool = config.mapping.verbose

        self.gamma = self.config.mapping.gamma
        self.cell_resolution = (self.axis_resolution, self.axis_resolution)
        self.num_iterations = 1
        
        self.cell_max_min = (self.world_bounds[0], self.world_bounds[1], self.world_bounds[0], self.world_bounds[1])

        self.grid = self.__calc_grid_auto()

        self.query_x: np.ndarray = np.arange(
            self.world_bounds[0],
            self.world_bounds[1] - 1,
            self.axis_resolution
        )
        self.query_y: np.ndarray = np.arange(
            self.world_bounds[0],
            self.world_bounds[1] - 1,
            self.axis_resolution
        )

        xx, yy = np.meshgrid(self.query_x, self.query_y)
        self.query_grid: np.ndarray = np.hstack((
            xx.ravel()[:, np.newaxis],
            yy.ravel()[:, np.newaxis]
        ))

        self.ogm: np.ndarray = np.ones((self.query_grid.shape[0], self.query_grid.shape[1])) * 0.5
        self.ogm = torch.from_numpy(self.ogm).to(self.device)




    def fit(self, X: List[np.ndarray], y: List[np.ndarray]) -> None:
        """
        Fit the VSA_OGM model to the given data.

        Args:
            X (List[np.ndarray]): The input data to fit the model to.
            y (List[np.ndarray]): The target data to fit the model to.

        Returns:
            None
        """
        fit_metrics: dict = {}

        if len(X) != len(y):
            raise ValueError("The number of input and target data must match.")
        
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)
            y = torch.tensor(y)

        if X.device != self.device:
            X = X.to(self.device)

        if y.device != self.device:
            y = y.to(self.device)

        y = y.view(-1, 1)
        
        start = time.time()

        if self.device.startswith("cuda"):
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        else:
            start_time = time.time()

        self.train(X, y)



        start = time.time()

        Y_query = self.predict_grid(self.query_grid)
        self.ogm = Y_query.cpu().numpy()
        self.ogm = np.reshape(
            self.ogm,                                        # (MxN,)
            (self.query_y.shape[0], self.query_x.shape[0])  # (M, N)
        )

        if self.device.startswith("cuda"):
            end_time.record()
            torch.cuda.synchronize()
            fit_metrics["training_time"] = start_time.elapsed_time(end_time)
        else:
            fit_metrics["training_time"] = (time.time() - start_time) / 1000

        if self.verbose:
            print(f"(SBHM) Training Time: {fit_metrics['training_time']}")

        self.num_observations += 1

        return fit_metrics, {}
    
    def train(self, X, y):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X)

        N, D = X.shape[0], X.shape[1]

        self.epsilon = torch.ones(N, dtype=torch.float32)
        if not hasattr(self, 'mu'):
            self.mu = torch.zeros(D, dtype=torch.float32)
            self.sig = 10000 * torch.ones(D, dtype=torch.float32)

        for i in range(1):
            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # print(f"MU SHAPE: {self.mu.shape}")
            # print(f"Sigma Shape: {self.sig.shape}")                

            # M-step
            # self.epsilon = torch.sqrt(torch.sum((X**2)*self.sig, dim=1) + (X.mm(self.mu.reshape(-1, 1))**2).squeeze())

        # print(self.mu)

        return self.mu, self.sig
    
    def predict_grid(self, Xq: np.ndarray) -> torch.Tensor:
        """
        :param Xq: raw inquery points
        :return: mean occupancy (Lapalce approximation)
        """

        if not isinstance(Xq, torch.Tensor):
            Xq = torch.from_numpy(Xq).to(self.device)

        Xq = self.__sparse_features(Xq)

        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        sig2_inv_a = torch.sum((Xq ** 2) * self.sig, dim=1)
        k = 1.0 / torch.sqrt(1 + np.pi * sig2_inv_a / 8)

        output = torch.sigmoid(k * mu_a)

        return output
    
    def predict(self, Xq: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict the output from the input data.

        Args:
            X (List[np.ndarray]): The input data to predict the output from.

        Returns:
            List[np.ndarray]: The predicted output data.
        """
        predictions: List[np.ndarray] = []
        prediction_metrics: dict = {}

        print(f"X Shape: {Xq.shape}")

        if not isinstance(Xq, torch.Tensor):
            Xq = torch.from_numpy(Xq).to(self.device)

        if Xq.device != self.device:
            Xq = Xq.to(self.device)

        Xq = self.__sparse_features(Xq)

        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        sig2_inv_a = torch.sum((Xq ** 2) * self.sig, dim=1)
        k = 1.0 / torch.sqrt(1 + np.pi * sig2_inv_a / 8)

        output = torch.sigmoid(k * mu_a)
        output = output.cpu().numpy()


        print(f"Predictions Shape: {output.shape}")

        return output, {}
    
    def reset(self) -> None:
        """
        Reset the observation space to its initial state. This is designed to
        be called alongside the reset method of the environment.

        Arguments:
        ----------
        None

        Returns:
        --------
        map : np.ndarray
            The initial observation space.
        """
        self.scan_no = 0

        if hasattr(self, "epsilon"):
            del self.epsilon

        if hasattr(self, "mu"):
            del self.mu

        if hasattr(self, "sig"):
            del self.sig

        Y_query = np.ones((self.query_grid.shape[0],)) * 0.5
        Y_query = np.reshape(
            Y_query,                                        # (MxN,)
            (self.query_y.shape[0], self.query_x.shape[0])  # (M, N)
        )

        return Y_query

    def __calc_grid_auto(self):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """

        x_min, x_max = self.cell_max_min[0], self.cell_max_min[1]
        y_min, y_max = self.cell_max_min[2], self.cell_max_min[3]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.cell_resolution[0]), \
                             np.arange(y_min, y_max, self.cell_resolution[1]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

        return torch.tensor(grid, dtype=torch.float32, device=self.device)
    
    def __calc_posterior(self, X, y, epsilon, mu0, sig0):
        """
        :param X: input features
        :param y: labels
        :param epsilon: per dimension local linear parameter
        :param mu0: mean
        :param sig0: variance
        :return: new_mean, new_varaiance
        """

        if X.device != self.device:
            X = X.to(self.device)

        if y.device != self.device:
            y = y.to(self.device)

        if epsilon.device != self.device:
            epsilon = epsilon.to(self.device)

        if mu0.device != self.device:
            mu0 = mu0.to(self.device)

        if sig0.device != self.device:
            sig0 = sig0.to(self.device)

        logit_inv = torch.sigmoid(epsilon)
        lam = 0.5 / epsilon * (logit_inv - 0.5)
        sig = 1/(1/sig0 + 2*torch.sum( (X.t()**2)*lam, dim=1))
        mu = sig*(mu0/sig0 + torch.mm(X.t(), y - 0.5).squeeze())
        return mu, sig

    def __rbf_kernel(self, X1, X2, gamma):
        K = torch.norm(X1[:, None] - X2, dim=-1, p=2).pow(2)
        K = torch.exp(-gamma*K)

        return K

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = self.__rbf_kernel(X, self.grid, gamma=self.gamma)
        rbf_features = torch.cat((torch.ones(X.shape[0],1).to(self.device), rbf_features), dim=1)

        return rbf_features
