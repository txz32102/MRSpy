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
    
    def get_chemical_shifts(self, n: int = 3):
        """
        For each dynamic t, extract the top n indices of the highest values from the summed spectra.
        
        :param n: Number of top indices to extract (default is 3)
        :return: A 2D array with shape (t, n) containing the top n indices for each dynamic.
        """
        # Get the shape of the data
        t, l, h, w = self.data.shape
        
        # Initialize an empty array to store the summed spectra with shape (t, l)
        summed_spectra = np.zeros((t, l))
        
        # Iterate over each dynamic t
        for i in range(t):
            # Reshape the data for the current dynamic to (l, h*w) and sum along the last axis
            # This will give the summed spectrum for the current dynamic
            summed_spectra[i] = np.sum(self.data[i].reshape(l, h * w), axis=1)
        
        # Initialize an array to store the top n indices for each dynamic
        top_indices = np.zeros((t, n), dtype=int)
        
        # Iterate over each summed spectrum to find the top n indices
        for i in range(t):
            # Get the indices of the top n values in descending order
            top_indices[i] = np.argsort(summed_spectra[i])[-n:][::-1]
        
        return top_indices
    
    
    def spec_plot(self, 
                  idx: int = 0, 
                  path=None, 
                  plot_all=False, 
                  xy=None, 
                  show_xy: bool = False,
                  annotate_top_n: bool = False,  # New parameter
                  n_dots: int = 3,  # Number of top points to annotate
                  title=None, xlabel=None, 
                  ylabel=None, 
                  xticks=None, 
                  **kwargs):
        """
        Plot time-series data from a 4D array (assuming self.data is that array) and save the plot if a path is specified.

        Data shape assumed: (num_subplots, num_dots, num_lines_dim1, num_lines_dim2) which is (26, 64, 32, 32) in your case.
        Plots 32x32 lines in each subplot, with each line having 64 dots.

        Parameters:
        - idx (int): Index to select a subplot (from the first dimension of self.data) when plot_all=False.
        - path (str, optional): Path to save the plot.
        - plot_all (bool): Plot all subplots as subplots in a single figure if True.
        - xy (tuple, optional): Grid size (m, n) for subplots when plot_all=True.
        - show_xy (bool): Show x-axis and y-axis values on each subplot (default: False).
        - annotate_top_n (bool): If True, annotate the top n maximum values on each subplot with dashed lines.
        - n (int): Number of top maximum values to annotate (default: 3).
        - title (str, optional): Title of the plot. Defaults to 'Subplot {subplot_index + 1}' or 'Subplot {idx + 1}'.
        - xlabel (str, optional): Label for the x-axis. Defaults to 'X-axis (0 to 64)'.
        - ylabel (str, optional): Label for the y-axis. Defaults to 'Y-axis'.
        - xticks (list or array-like, optional): X-axis tick positions. Defaults to range(0, 65, 5).
        - **kwargs: Additional arguments for `plt.savefig`.
        """
        data_array = self.data

        if plot_all:
            n_subplots = data_array.shape[0]
            if xy is None:
                m = int(np.ceil(np.sqrt(n_subplots)))
                n = int(np.ceil(n_subplots / m))
            else:
                m, n = xy
                if m * n < n_subplots:
                    raise ValueError("The provided grid size (m, n) is too small for the input data.")

            fig, axs = plt.subplots(m, n, figsize=(n * 5, m * 4))
            axs = axs.ravel()

            for subplot_index in range(n_subplots):
                ax = axs[subplot_index]

                # Plot all lines
                for line_dim1_index in range(data_array.shape[2]):  # 32 lines dimension 1
                    for line_dim2_index in range(data_array.shape[3]):  # 32 lines dimension 2
                        y_values = data_array[subplot_index, :, line_dim1_index, line_dim2_index]
                        x_values = range(data_array.shape[1])  # 64 dots
                        ax.plot(x_values, y_values, linewidth=0.5, alpha=0.5)

                    if annotate_top_n:
                        # Sum along lines dimensions to get the summed spectrum
                        summed_spectrum = np.sum(data_array[subplot_index], axis=(1, 2))
                        # Get top n indices
                        top_n_indices = np.argsort(summed_spectrum)[-n_dots:][::-1]

                        # For each top n index, find the maximum y-value among all lines at that index
                        for idx in top_n_indices:
                            # Get the maximum y-value at the current index across all lines
                            max_y_value = np.max(data_array[subplot_index, idx])
                            # Mark x position with a vertical dashed line
                            ax.axvline(x=idx, color='r', linestyle='--', linewidth=1)
                            # Mark y position with a horizontal dashed line
                            ax.axhline(y=max_y_value, color='r', linestyle='--', linewidth=1)

                # Use provided titles/labels or defaults
                ax_title = title if title is not None else f'Subplot {subplot_index + 1}'
                ax_xlabel = xlabel if xlabel is not None else f'X-axis (0 to {self.data.shape[1]})'
                ax_ylabel = ylabel if ylabel is not None else 'Y-axis'
                ax_xticks = xticks if xticks is not None else range(0, self.data.shape[1], 5)

                ax.set_title(ax_title)
                ax.set_xlabel(ax_xlabel)
                ax.set_ylabel(ax_ylabel)
                ax.set_xticks(ax_xticks)
                ax.grid(True, linestyle='--', alpha=0.6)

                if not show_xy:
                    ax.axis('off')  # Turn off axes if show_xy is False

            # Remove any unused subplots if grid is larger than needed
            for i in range(n_subplots, len(axs)):
                fig.delaxes(axs[i])

            plt.tight_layout()
            if path:
                plt.savefig(path, **kwargs)
            plt.show()

        else:  # plot_all is False, plot a single subplot
            temp_data = data_array[idx]  # Select single subplot data
            fig, ax = plt.subplots(figsize=(10, 8))  # Create figure and axis objects

            # Plot all lines
            for line_dim1_index in range(temp_data.shape[1]):  # 32 lines in dimension 1
                for line_dim2_index in range(temp_data.shape[2]):  # 32 lines in dimension 2
                    y_values = temp_data[:, line_dim1_index, line_dim2_index]
                    x_values = range(temp_data.shape[0])  # 64 dots
                    ax.plot(x_values, y_values, linewidth=0.5, alpha=0.5)  # Plot using ax

            # Annotate top n values if required
            if annotate_top_n:
                summed_spectrum = np.sum(temp_data, axis=(1, 2))  # Sum along lines dimensions
                top_n_indices = np.argsort(summed_spectrum)[-n_dots:][::-1]  # Get top n indices
                for idx in top_n_indices:
                    ax.axvline(x=idx, color='r', linestyle='--', linewidth=1)  # Mark x positions
                    ax.axhline(y=summed_spectrum[idx], color='r', linestyle='--', linewidth=1)  # Mark y positions

            # Use provided titles/labels or defaults
            ax_title = title if title is not None else f'Subplot {idx + 1}'
            ax_xlabel = xlabel if xlabel is not None else f'X-axis (0 to {self.data.shape[1]})'
            ax_ylabel = ylabel if ylabel is not None else 'Y-axis'
            ax_xticks = xticks if xticks is not None else range(0, self.data.shape[1], 5)

            ax.set_title(ax_title)
            ax.set_xlabel(ax_xlabel)
            ax.set_ylabel(ax_ylabel)
            ax.set_xticks(ax_xticks)
            ax.grid(True, linestyle='--', alpha=0.6)

            if not show_xy:
                ax.axis('off')  # Turn off axes if show_xy is False

            plt.tight_layout()
            if path:
                plt.savefig(path, **kwargs)
            plt.show()
            
class TensorComparePlotter:
    def __init__(self, tensor1=None, tensor2=None):
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    @classmethod
    def from_tensors(cls, tensor1, tensor2):
        tensor1 = cls._convert_to_numpy(tensor1)
        tensor2 = cls._convert_to_numpy(tensor2)
        if tensor1.shape != tensor2.shape:
            raise ValueError("Tensors must have the same shape.")
        return cls(tensor1, tensor2)

    @classmethod
    def from_paths(cls, path1, path2):
        data1 = np.load(path1)  # Assuming load_mrs_mat uses np.load or similar
        data2 = np.load(path2)
        return cls(data1, data2)

    @staticmethod
    def _convert_to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        raise TypeError("Tensors must be numpy arrays or PyTorch tensors")

    def compare_plot(self, t_index=0, path=None, show_xy=True,
                     title=None, xlabel=None, ylabel=None,
                     xticks=None, **kwargs):
        """
        Plots comparative data from two tensors for a fixed 't' index.

        Parameters:
            t_index (int): Index of the time dimension to fix and plot.
            path (str, optional): Path to save the plot.
            show_xy (bool): Toggle axis visibility.
            title (str): Main title for the plot.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            xticks (list): Custom x-axis ticks.
            **kwargs: Additional arguments for `plt.savefig`.
        """
        if (t_index < 0 or t_index >= self.tensor1.shape[0] or
            t_index >= self.tensor2.shape[0]):
            raise IndexError("t_index out of bounds")

        data1 = self.tensor1[t_index]
        data2 = self.tensor2[t_index]
        l = data1.shape[0]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs = axs.ravel()

        common_params = {
            'xlabel': xlabel or f'',
            'ylabel': ylabel or '',
            'xticks': xticks or range(0, l, max(l//10, 1) if l > 10 else 1)
        }

        for ax, title_part, data in zip(axs, ['Simulated', 'Real'], [data1, data2]):
            for w in range(data.shape[1]):
                for h in range(data.shape[2]):
                    ax.plot(np.arange(l), data[:, w, h], lw=0.5, alpha=0.5)

            ax.set_title(f'{title_part} Data')
            ax.set_xlabel(common_params['xlabel'])
            ax.set_ylabel(common_params['ylabel'])
            ax.set_xticks(common_params['xticks'])
            ax.grid(True, linestyle='--', alpha=0.6)

            if not show_xy:
                ax.tick_params(labelbottom=False, labelleft=False,
                               bottom=False, left=False)

        if title:
            fig.suptitle(title)

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
    axis: Optional[str] = 'none',  # Change default to 'none'
    path: Optional[str] = None, 
    dpi: int = 100
) -> None:
    """
    Plots a numpy array or a torch tensor as an image.

    Parameters:
    - data: numpy array or torch tensor (CPU/GPU, with or without gradients), shape (w, h) or (w, h, c)
    - axis: 'none' to hide axes, any other value to show them (default: 'none')
    - path: optional path to save the figure (default: None)
    - dpi: resolution of the saved figure (default: 100)
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()  # Move to CPU and detach if necessary
    
    if data.ndim == 2:
        plt.figure()
        plt.imshow(data, cmap='gray')
    
    elif data.ndim == 3:
        if data.shape[-1] == 1:
            plt.figure()
            plt.imshow(data[..., 0], cmap='gray')
        else:
            plt.figure()
            plt.imshow(data)
    
    else:
        raise ValueError("Input data must have shape (w, h) or (w, h, c)")

    if axis == 'none':
        plt.axis('off')
    else:
        plt.axis('on')  # Show axes for any other value, including None

    if path:
        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()

def plot_3d_array(data: Union[np.ndarray, torch.Tensor], 
                  path: str = None, 
                  save_path: str = None, 
                  dpi: int = 300, 
                  cbar: bool = False, 
                  cmap: str = 'hot', 
                  show_xy: bool = False, 
                  normalize_range: bool = False):
    """
    Plots a 3D array or tensor of shape (b, w, h) as a 1 * b subplot.

    Parameters:
        data (np.ndarray or torch.Tensor): The input 3D array or tensor with shape (b, w, h).
        path (str): Alternative parameter for specifying the save path.
        save_path (str): Path to save the image. If not provided, the image will be displayed.
        dpi (int): Resolution for saving the image, default is 300.
        cbar (bool): Whether to include a color bar, default is False.
        cmap (str): Colormap to use for visualization, default is 'hot'.
        show_xy (bool): If True, shows the x-y axis on each subplot.
        normalize_range (bool): If True, normalizes the color scale across the entire data range.
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

    # Determine the normalization range if needed
    vmin, vmax = (None, None)
    if normalize_range:
        vmin = np.min(data)
        vmax = np.max(data)

    # Iterate through each subplot and plot the data
    for i in range(b):
        # Use the specified colormap to plot each 2D array
        ax = axes[i] if b > 1 else axes  # If b=1, axes is not an array
        im = ax.imshow(data[i], cmap=cmap, vmin=vmin, vmax=vmax) 
        if not show_xy:
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