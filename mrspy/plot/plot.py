import numpy as np
import matplotlib.pyplot as plt
import typing
from math import ceil
import os

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