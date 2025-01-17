import numpy as np
import matplotlib.pyplot as plt
import typing
from math import ceil
import os
from mrspy.util import load_mrs_mat
import torch

class SpecPlotter:
    def __init__(self, data: typing.Optional[np.ndarray] = None):
        self.data = data

    @classmethod
    def from_tensor(cls, tensor: typing.Union[np.ndarray, "torch.Tensor"]):
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

    def time(self, idx: int = 0, saved_plot=None, **kwargs):
        # temp is of shape (120, 32, 32)
        temp = self.data[idx]
        
        # Create a figure
        plt.figure(figsize=(10, 6))
        
        for i in range(temp.shape[1]):
            for j in range(temp.shape[2]):
                plt.plot(temp[:, i, j])
        plt.tight_layout()
        if(saved_plot):
            plt.savefig(saved_plot, **kwargs)
        plt.show()


def plot_chemicalshift_image(data: np.ndarray, chemicalshift: typing.List[int]=[67, 61, 49], cmap: str = 'hot', path="./", dpi=600):
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
        plt.savefig(f'{path}/plot_{ichem}.jpg', dpi=dpi)
        plt.close(fig)
