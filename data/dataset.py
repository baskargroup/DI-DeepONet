import torch
from torch.utils.data import Dataset
import numpy as np

class LidDrivenDataset3D(Dataset):
    """
    Custom dataset for loading and processing Lid Driven Cavity problem data from .npz files.
    """
    def __init__(self, file_path_x, file_path_y, data_type=None, geometric_deeponet=False):
        """
        Initializes the dataset with the paths to the .npz files and processes the data.

        Args:
            file_path_x (str): Path to the .npz file containing the input data.
            file_path_y (str): Path to the .npz file containing the target data.
            equation (str): Type of equation ("ns" or "ns+ht").
            data_type (str): Type of data ("field" or "collocation").
            inputs (str): Input type ("sdf" or "mask").
            geometric_deeponet (bool): Whether to use geometric DeepONet configuration.
        """
        # Load data from .npz files
        x = np.load(file_path_x)['data']  # Input data [num_samples, [Re, SDF], res, res, res]
        y = np.load(file_path_y)['data']  # Target data [num_samples, [u, v, w], res, res, res]

        #if not geometric_deeponet:
        x = x[:, :2, :, :, :]  # Keep [Re, SDF]
        y = y[:, :3, :, :, :]  # Keep [u, v, w]


        if data_type == 'field':
            self.collocation = False
        elif data_type == 'collocation':
            self.collocation = True
            self.resolution = x.shape[-1]

        # Convert numpy arrays to PyTorch tensors
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the input and target tensors at the specified index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            tuple: (input_tensor, target_tensor)
        """
        sample = self.x[idx]
        target = self.y[idx]

        if self.collocation:
            grid = self.get_grid()
            sample = torch.cat((sample, grid), dim=0)

        return sample, target

    def get_grid(self):
        """
        Generates a uniform grid for collocation points in 3D.

        Returns:
            torch.Tensor: A grid tensor with shape (3, resolution, resolution, resolution).
        """
        # Create the uniform grid for x, y, and z locations of the input
        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(0, 1, self.resolution), 
            np.linspace(0, 1, self.resolution),
            np.linspace(0, 1, self.resolution),
            indexing='ij'  # Use 'ij' indexing for compatibility with tensor operations
        )
        # Stack the grids along the last dimension to get (res, res, res, 3) shape
        grid = np.stack([grid_x, grid_y, grid_z], axis=-1)
        # Transpose to get shape (3, res, res, res) instead of (res, res, res, 3)
        grid = grid.transpose(3, 0, 1, 2)
        # Convert to PyTorch tensor
        grid = torch.tensor(grid, dtype=torch.float)
        return grid
