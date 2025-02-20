import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio

class datapipeline():
    def __init__(self,               
                CSF_file_path,
                GM_file_path,
                WM_file_path,
                input_type=".nii.gz", 
                weight_dict={
                    "CSF": [0.1, 0.3, 0.6],
                    "GM": [0.3, 0.6, 0.2],
                    "WM": [0.6, 0.1, 0],
                },
                size=None,
                slice_dim=0, 
                pad=0.0,
                pad_axis="y",
                no_weight_dict=False):
        self.CSF_file_path = CSF_file_path
        self.GM_file_path = GM_file_path
        self.WM_file_path = WM_file_path
        self.input_type = input_type
        self.weight_dict = weight_dict
        self.size = size
        self.slice_dim = slice_dim
        self.pad = pad
        self.pad_axis = pad_axis
        self.water = None
        self.glu = None
        self.lac = None
        
    def load_data(self):
        if(self.input_type == ".nii.gz"):
            csf_img  = nib.load(self.CSF_file_path).get_fdata()
            gm_img = nib.load(self.GM_file_path).get_fdata()
            wm_img = nib.load(self.WM_file_path).get_fdata()
            
            # water = CSF * 0.1 + GM * 0.3 + WM * 0.6;
            # glu = CSF * 0.3 + GM * 0.6 + WM * 0.1;
            # lac = CSF * 0.6 + GM * 0.2 + WM * 0;
            self.water = self.weight_dict["CSF"][0] * csf_img + self.weight_dict["GM"][0] * gm_img + self.weight_dict["WM"][0] * wm_img
            self.glu = self.weight_dict["CSF"][1] * csf_img + self.weight_dict["GM"][1] * gm_img + self.weight_dict["WM"][1] * wm_img
            self.lac = self.weight_dict["CSF"][2] * csf_img + self.weight_dict["GM"][2] * gm_img + self.weight_dict["WM"][2] * wm_img
    
    def process(self):
        if(self.water is None and self.glu is None and self.lac is None):
            self.load_data()
            
        if(self.pad is not None):
            self.water = self._pad_3d(self.water)
            self.glu = self._pad_3d(self.glu)
            self.lac = self._pad_3d(self.lac)
        if(self.size is not None):
            self.water = self._resize_3d(self.water)
            self.glu = self._resize_3d(self.glu)
            self.lac = self._resize_3d(self.lac)

    def _pad_3d(self, arr):
        # Get the shape of the input array
        shape = arr.shape

        # Determine the padding size based on pad_axis and pad
        if self.pad_axis == "x":
            pad_size = int(shape[2] * self.pad)  # Padding along the x-axis (third dimension)
        elif self.pad_axis == "y":
            pad_size = int(shape[1] * self.pad)  # Padding along the y-axis (second dimension)
        else:
            raise ValueError("pad_axis must be 'x' or 'y'")

        # Calculate the padding widths for each dimension
        pad_widths = [(0, 0), (0, 0), (0, 0)]  # Initialize padding widths for three dimensions

        if self.pad_axis == "x":
            pad_widths[2] = (pad_size // 2, pad_size - pad_size // 2)  # Padding for x-axis
        elif self.pad_axis == "y":
            pad_widths[1] = (pad_size // 2, pad_size - pad_size // 2)  # Padding for y-axis

        # Apply padding using numpy.pad
        padded_arr = np.pad(arr, pad_width=pad_widths, mode='constant', constant_values=0)

        return padded_arr
    
    def _resize_3d(self, arr):
        slice_dim = self.slice_dim
        original_shape = arr.shape
        w, h = self.size
        
        if slice_dim == 0:
            resized_arr = np.zeros((original_shape[0], w, h), dtype=arr.dtype)
            for i in range(original_shape[0]):
                resized_arr[i, :, :] = cv2.resize(arr[i, :, :], (h, w))
        elif slice_dim == 1:
            resized_arr = np.zeros((w, original_shape[0], h), dtype=arr.dtype)
            for i in range(original_shape[1]):
                resized_arr[:, i, :] = cv2.resize(arr[:, i, :], (h, w))
        elif slice_dim == 2:
            resized_arr = np.zeros((w, h, original_shape[0]), dtype=arr.dtype)
            for i in range(original_shape[2]):
                resized_arr[:, :, i] = cv2.resize(arr[:, :, i], (h, w))
        else:
            raise ValueError("Invalid slice_dim! It must be 0, 1, or 2.")

        return resized_arr
    

    def save(self, output_path, idx_range=None, output_type="mat"):
        if idx_range is None:
            shape = self.water.shape
            start = shape[self.slice_dim] * 0.6
            idx_range = [int(start), int(start + 10)]
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Determine the slice indices
        start_idx, end_idx = idx_range
        
        # Loop through the slices and save the data
        for cur_slice in range(start_idx, end_idx):
            # Create folder for each slice
            slice_folder = os.path.join(output_path, str(cur_slice))
            if not os.path.exists(slice_folder):
                os.makedirs(slice_folder)
            
            # Get the sliced data based on the slice_dim
            if self.slice_dim == 0:
                water_data = self.water[cur_slice, :, :]
                glu_data = self.glu[cur_slice, :, :]
                lac_data = self.lac[cur_slice, :, :]
            elif self.slice_dim == 1:
                water_data = self.water[:, cur_slice, :]
                glu_data = self.glu[:, cur_slice, :]
                lac_data = self.lac[:, cur_slice, :]
            elif self.slice_dim == 2:
                water_data = self.water[:, :, cur_slice]
                glu_data = self.glu[:, :, cur_slice]
                lac_data = self.lac[:, :, cur_slice]
            
            if output_type == "mat":
                water_data = np.array(water_data, dtype=np.float32)
                glu_data = np.array(glu_data, dtype=np.float32)
                lac_data = np.array(lac_data, dtype=np.float32)
                # Save as .mat files
                sio.savemat(os.path.join(slice_folder, "water.mat"), {'water': water_data})
                sio.savemat(os.path.join(slice_folder, "glu.mat"), {'glu': glu_data})
                sio.savemat(os.path.join(slice_folder, "lac.mat"), {'lac': lac_data})
            elif output_type in ["png", "jpg"]:
                # Normalize the data to [0, 255] for saving as images
                water_data = np.uint8(np.clip(water_data * 255.0, 0, 255))
                glu_data = np.uint8(np.clip(glu_data * 255.0, 0, 255))
                lac_data = np.uint8(np.clip(lac_data * 255.0, 0, 255))
                
                # Save as .png or .jpg files using OpenCV
                cv2.imwrite(os.path.join(slice_folder, f"water.{output_type}"), water_data)
                cv2.imwrite(os.path.join(slice_folder, f"glu.{output_type}"), glu_data)
                cv2.imwrite(os.path.join(slice_folder, f"lac.{output_type}"), lac_data)

    
    def plot(self, slice_dim=None, idx=0, cmap='hot', save_path=None, dpi=100):
        if slice_dim is None:
            slice_dim = self.slice_dim

        if slice_dim == 0:
            water_slice = self.water[idx, :, :]
            glu_slice = self.glu[idx, :, :]
            lac_slice = self.lac[idx, :, :]
        elif slice_dim == 1:
            water_slice = self.water[:, idx, :]
            glu_slice = self.glu[:, idx, :]
            lac_slice = self.lac[:, idx, :]
        elif slice_dim == 2:
            water_slice = self.water[:, :, idx]
            glu_slice = self.glu[:, :, idx]
            lac_slice = self.lac[:, :, idx]
        else:
            raise ValueError("slice_dim must be 0, 1, or 2")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(water_slice, cmap=cmap)
        im1 = axes[1].imshow(glu_slice, cmap=cmap)
        im2 = axes[2].imshow(lac_slice, cmap=cmap)

        axes[0].set_title('Water')
        axes[1].set_title('Glu')
        axes[2].set_title('Lac')
        
        cbar = plt.colorbar(im2, ax=axes, orientation='vertical')
        fig.subplots_adjust(right=0.75) 
        if save_path:
            plt.savefig(save_path, dpi=dpi)

        plt.show()
