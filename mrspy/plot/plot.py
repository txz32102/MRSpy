import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
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
                axs[idx].axis('off') 
            
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

def plot_chemicalshift_image(data: np.ndarray, chemicalshift: List[int] = [67, 61, 49], 
                             cbar: bool = False, cmap: str = 'hot', path="./", dpi=600, 
                             order: List[str] = None, normalize_range: bool = False, no_gap: bool = False):
    """
    Plots images for chemical shifts from the provided tensor data.

    Parameters:
    - data: A tensor-like object, will be converted to a numpy array.
    - chemicalshift: List of indices corresponding to chemical shifts.
    - cbar: If True, adds a colorbar to each subplot.
    - cmap: Colormap to use for the images.
    - path: Directory to save the images.
    - dpi: Resolution of the saved images.
    - order: Optional list of filenames for the saved images.
    - normalize_range: If True, normalizes the color scale across the entire data range.
    - no_gap: If True, removes all gaps between subplots.

    Saves images named `gt_<chemical shift index>.jpg` or as specified by `order`.
    """

    os.makedirs(path, exist_ok=True)
    data = np.array(data)  # Ensure data is a numpy array
    
    for ichem, shift in enumerate(chemicalshift):
        image_gd = data[:, shift, :, :]

        n_rows = ceil(image_gd.shape[0] / 8)
        if no_gap:
            # Set figure size to match the number of subplots exactly (assuming square images)
            fig, axes = plt.subplots(n_rows, 8, figsize=(8, n_rows))
        else:
            fig, axes = plt.subplots(n_rows, 8, figsize=(16, 10))
        axes = axes.flatten()

        for iNum in range(image_gd.shape[0]):
            ax = axes[iNum]
            if normalize_range:
                im = ax.imshow(np.abs(image_gd[iNum, :, :]), cmap=cmap, vmin=np.min(data), vmax=np.max(data))
            else:
                im = ax.imshow(np.abs(image_gd[iNum, :, :]), cmap=cmap)
            ax.axis('off')

            # Add colorbar if cbar is True
            if cbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Disable unused axes
        for i in range(image_gd.shape[0], len(axes)):
            axes[i].axis('off')

        if no_gap:
            # Remove all gaps and padding
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        else:
            plt.tight_layout()

        if order is not None:
            plt.savefig(f'{path}/{order[ichem]}.jpg', dpi=dpi)
        else:
            plt.savefig(f'{path}/plot_{ichem}.jpg', dpi=dpi)
        plt.close(fig)
        
def plot(
    data: Union[np.ndarray, torch.Tensor], 
    axis: Optional[str] = None, 
    path: Optional[str] = None, 
    dpi: int = 100
) -> None:
    """
    Plots a numpy array or a torch tensor as an image.

    Parameters:
    - data: numpy array or torch tensor (CPU/GPU, with or without gradients), shape (w, h)
    - axis: 'none' to hide axes, any other value to show them (default: None)
    - path: optional path to save the figure (default: None)
    - dpi: resolution of the saved figure (default: 100)
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()  # Move to CPU and detach if necessary
    
    if data.ndim != 2:
        raise ValueError("Input data must have shape (w, h)")
    
    plt.figure()
    plt.imshow(data, cmap='gray')
    
    if axis == 'none':
        plt.axis('off')
    
    if path:
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()

def plot_3d_array(data: Union[np.ndarray, torch.Tensor], path: str = None, save_path: str = None, dpi: int = 300, cbar: bool = False, cmap: str = 'hot'):
    """
    Plots a 3D array or tensor of shape (b, w, h) as a 1 * b subplot.

    Parameters:
        data (np.ndarray or torch.Tensor): The input 3D array or tensor with shape (b, w, h).
        path (str): Alternative parameter for specifying the save path.
        save_path (str): Path to save the image. If not provided, the image will be displayed.
        dpi (int): Resolution for saving the image, default is 300.
        cbar (bool): Whether to include a color bar, default is False.
        cmap (str): Colormap to use for visualization, default is 'hot'.
    """
    # Use save_path if provided; otherwise, fall back to path
    save_path = save_path or path

    # If input is a torch.Tensor, detach and convert it to a numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()  # Ensure it's on CPU and convert to numpy array

    # Check the shape of the input array
    if len(data.shape) != 3:
        raise ValueError("Input array must be 3D with shape (b, w, h).")
    
    b, w, h = data.shape  # Get batch size and image dimensions

    # Create a 1 × b subplot layout
    fig, axes = plt.subplots(1, b, figsize=(b * 2, 2))  # Each subplot width is 2 inches

    # Iterate through each subplot and plot the data
    for i in range(b):
        # Use the specified colormap to plot each 2D array
        ax = axes[i] if b > 1 else axes  # If b=1, axes is not an array
        im = ax.imshow(data[i], cmap=cmap) 
        ax.axis('off')  # Turn off the axis
        if cbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) 

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save or display the image
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"Image saved to {save_path}")
    else:
        plt.show()

    # Close the figure window
    plt.close(fig)
    
def plot_image_grid(
    data: Union[np.ndarray, torch.Tensor],
    cmap: Optional[str] = None,
    cbar: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300,
    xy: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot a grid of images from a 3D or 4D array (numpy array or PyTorch tensor).
    Supports both grayscale (b, w, h) and RGB (b, w, h, c) images.

    Parameters:
    - data: Union[np.ndarray, torch.Tensor]
        The input data with shape (b, w, h) for grayscale or (b, w, h, c) for RGB.
        Can be a numpy array or a PyTorch tensor.
    - cmap: Optional[str] (default=None)
        The colormap to use for grayscale images. Ignored for RGB images.
        If None, 'gray' will be used for grayscale images.
    - cbar: bool (default=False)
        Whether to add a colorbar to each subplot (only applicable for grayscale).
    - save_path: Optional[str] (default=None)
        Path to save the figure. If None, the figure will not be saved.
    - dpi: int (default=300)
        The resolution of the saved figure.
    - xy: Optional[Tuple[int, int]] (default=None)
        A tuple (m, n) specifying the number of rows and columns in the subplot grid.
        If not provided, the function will automatically determine the grid size.
    """
    # Convert PyTorch tensor to numpy array if necessary
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()  # Ensure it's on CPU and detached from computation graph

    # Validate data shape
    if len(data.shape) not in [3, 4]:
        raise ValueError("Input data must have shape (b, w, h) for grayscale or (b, w, h, c) for RGB.")

    is_rgb = len(data.shape) == 4
    if is_rgb:
        b, w, h, c = data.shape
        if c not in [3, 4]:  # Check for RGB or RGBA
            raise ValueError("RGB data must have 3 or 4 channels (RGB or RGBA).")
    else:
        b, w, h = data.shape

    # Determine grid size if not provided
    if xy is None:
        m = int(np.ceil(np.sqrt(b)))  # Number of rows
        n = int(np.ceil(b / m))       # Number of columns
    else:
        m, n = xy
        if m * n < b:
            raise ValueError("The provided grid size (m, n) is too small for the input data.")

    # Create figure and subplots
    fig, axes = plt.subplots(m, n, figsize=(n * 2, m * 2))
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    # Plot each image
    for i in range(b):
        if is_rgb:
            # For RGB images, no colormap is needed
            axes[i].imshow(data[i])
        else:
            # For grayscale, use provided cmap or default to 'gray'
            im = axes[i].imshow(data[i], cmap=cmap if cmap else 'gray')
            if cbar:
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        axes[i].axis('off')  # Turn off axis for better visualization

    # Hide remaining subplots if b < m * n
    for i in range(b, m * n):
        axes[i].axis('off')

    # Adjust layout and save figure if needed
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()