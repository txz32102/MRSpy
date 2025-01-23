import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List
from math import ceil
import os
from mrspy.util import load_mrs_mat
import torch

class SpecPlotter:
    def __init__(self, data: Optional[np.ndarray] = None):
        self.data = data

    @classmethod
    def from_tensor(cls, tensor: Union[np.ndarray, "torch.Tensor"]):
        if torch and isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        elif not isinstance(tensor, np.ndarray):
            raise TypeError("Input must be a NumPy array or PyTorch tensor.")
        
        return cls(data=tensor)

    @classmethod
    def from_path(cls, path: str):
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The file at path {path} does not exist.")
        data = load_mrs_mat(path)
        return cls(data=data)

    def spec_plot(self, idx: int = 0, path=None, plot_all=False, **kwargs, ):
        """
        Plot time-series data from a 3D array and save the plot if a path is specified.

        Parameters:
        - idx (int): Index to select a 3D array from self.data.
        - path (str, optional): Path to save the plot.
        - saved_plot (str, optional): Deprecated; use `path` instead.
        - **kwargs: Additional arguments for `plt.savefig`.

        Returns:
        - None
        """
        if plot_all == True:
            n_plots = self.data.shape[0]
            
            m = int(np.ceil(np.sqrt(n_plots)))
            n = int(np.ceil(n_plots / m))
            
            fig, axs = plt.subplots(m, n, figsize=(n * 2, m * 2))
            axs = axs.ravel()
            
            for idx, data in enumerate(self.data):
                if idx >= n_plots:
                    break
                axs[idx].plot(data.ravel())
            
            for i in range(idx + 1, len(axs)):
                fig.delaxes(axs[i])
            
            plt.tight_layout()
            
            if path:
                plt.savefig(path, **kwargs)
            
            plt.show()
        else:
            temp = self.data[idx]

            # Create a figure
            plt.figure(figsize=(10, 6))

            for i in range(temp.shape[1]):
                for j in range(temp.shape[2]):
                    plt.plot(temp[:, i, j])

            plt.tight_layout()

            if path:
                plt.savefig(path, **kwargs)

            plt.show()


def plot_chemicalshift_image(data: np.ndarray, chemicalshift: List[int]=[67, 61, 49], cmap: str = 'hot', path="./", dpi=600, order: List[str]=None):
    """
    Plots images for chemical shifts from the provided tensor data.

    Parameters:
    - data: A tensor-like object, will be converted to a numpy array.
    - chemicalshift: List of indices corresponding to chemical shifts.

    Saves images named `gt_<chemical shift index>.jpg`.
    """
    os.makedirs(path, exist_ok=True)
    data = np.array(data)  # Ensure data is a numpy array
    for ichem, shift in enumerate(chemicalshift):
        image_gd = data[:, shift, :, :]

        n_rows = ceil(image_gd.shape[0] / 8)
        fig, axes = plt.subplots(n_rows, 8, figsize=(16, 10))
        axes = axes.flatten()

        for iNum in range(image_gd.shape[0]):
            ax = axes[iNum]
            ax.imshow(np.abs(image_gd[iNum, :, :]), cmap=cmap)
            ax.axis('off')

        # Disable unused axes
        for i in range(image_gd.shape[0], len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        if(order is not None):
            plt.savefig(f'{path}/{order[ichem]}.jpg', dpi=dpi)
        else:
            plt.savefig(f'{path}/plot_{ichem}.jpg', dpi=dpi)
        plt.close(fig)
