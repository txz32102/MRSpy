import torch
from typing import Union
import numpy as np
from mrspy.util import load_mrs_mat, resize_image_batch, fft_xspace_to_kspace_3d_batch, fft_kspace_to_xspace_3d_batch
import os
from mrspy.util.noise import add_gaussian_noise
from mrspy.util.blur import update_averaging
import torch.nn.functional as F

class Simulation():
    def __init__(self, dce_number: int = 26, spec_len: int = 64, target_size: int = 32, cfg: dict = None):
        self.dce_number = dce_number
        self.spec_len = spec_len
        self.target_size = target_size
        self.chemical_dce_curve = None
        self.chemical_density_map = None
        if(cfg is None):
            self.cfg = {
                "curve": "default",
                "device": "cuda:0",
                "return_type": {
                    "gt",
                    "no",
                    "wei",
                    "wei_no",
                    "standard_mean",
                    "standard_sum"},
                "dtype": "float",
                "wei_no": {
                    "noise_level": 0.02
                },
                "no": {
                    "noise_level": 0.02
                },
                "wei": {
                    "average": 263
                },
                "return_dict" : True
            }
        else:
            self.cfg = cfg
        self.device = self.cfg.get('device')
        self.dtype = self.cfg.get('dtype')
        self._set_dtype()
        self.tackle_default_cfg()
        
    def tackle_default_cfg(self):
        if self.cfg.get("average") is None:
            self.cfg['average'] = 263
        
    def _set_dtype(self):
        if self.dtype == "float":
            self.torch_dtype = torch.float32
            self.complex_dtype = torch.complex64
        elif self.dtype == "double":
            self.torch_dtype = torch.float64
            self.complex_dtype = torch.complex64
        elif self.dtype == "half":
            self.torch_dtype = torch.float16
            self.complex_dtype = torch.complex64
        else:
            raise ValueError("dtype must be 'float', 'double', or 'half'")
        
    @torch.no_grad()
    def load_curve(self, curve: Union[str, torch.Tensor]) -> None:
        if isinstance(curve, torch.Tensor):
            self.chemical_dce_curve = curve.to(dtype=self.torch_dtype, device=self.device)
        else:
            self.chemical_dce_curve = load_mrs_mat(curve, output_type="tensor", dtype=self.torch_dtype, device=self.device)
        
    @torch.no_grad()
    def load_default_curve(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "../data", "gray_dynamic_curve.mat")
        data_path = os.path.normpath(data_path)
        self.chemical_dce_curve = load_mrs_mat(data_path, output_type="tensor", 
                                            dtype=self.torch_dtype, device=self.device)
        
        column_indices = torch.arange(0, 52, 2)
        water_dy = self.chemical_dce_curve[2, column_indices]
        glu_dy = self.chemical_dce_curve[0, column_indices]
        lac_dy = self.chemical_dce_curve[1, column_indices]
        self.chemical_dce_curve = torch.stack((water_dy, glu_dy, lac_dy), dim=0).to(self.device)
    
    @torch.no_grad()
    def generate_chemical_density_map(self, components: torch.Tensor):
        """
        Generate chemical density map for N components.
        Input: components - tensor of shape (B, N, H, W)
        Output: self.chemical_density_map - tensor of shape (B, N, DCE, H, W)
        """
        B, N, H, W = components.shape
        assert N == self.chemical_dce_curve.shape[0], "Number of components must match the number of curves"
        
        # Take absolute value and expand to DCE dimension
        components_abs = torch.abs(components)  # (B, N, H, W)
        components_exp = components_abs.unsqueeze(2).expand(-1, -1, self.dce_number, -1, -1)  # (B, N, DCE, H, W)
        
        # Expand curves for broadcasting
        curves = self.chemical_dce_curve[:, :self.dce_number].view(1, N, self.dce_number, 1, 1)  # (1, N, DCE, 1, 1)
        
        # Compute density map using broadcasting
        self.chemical_density_map = components_exp * curves  # (B, N, DCE, H, W)

    @torch.no_grad()
    def simulation(self, components: torch.Tensor, abs=True):
        """
        Perform MRSI simulation for N components.
        Input: components - tensor of shape (B, N, H, W)
        Output: Dictionary with simulation results
        """
        # Ensure input is a tensor on the correct device
        B, N, H, W = components.shape
        if isinstance(components, np.ndarray):
            components = torch.tensor(components, dtype=self.torch_dtype, device=self.device)
        else:
            components = components.to(self.device).to(self.torch_dtype)
        
        # Resize components to target size
        components_resized = F.interpolate(components, size=(self.target_size, self.target_size), 
                                         mode='bilinear', align_corners=False)  # (B, N, target_size, target_size)
        
        # Load dynamic curves
        curve = self.cfg.get("curve")
                
        if isinstance(curve, str):
            if(curve == "default" or curve is None):
                self.load_default_curve()  # Loads curves for 3 components
        else:
            self.chemical_dce_curve = torch.tensor(curve, dtype=self.torch_dtype, device=self.device)
        
        # Generate density maps
        self.generate_chemical_density_map(components_resized)
        
        # Set number of components
        chemical_com = N
        
        # Define simulation parameters
        if "chemical_shifts" in self.cfg:
            chemical_shift = torch.tensor(self.cfg["chemical_shifts"], dtype=self.torch_dtype, device=self.device)
            assert chemical_shift.shape[0] == N, "chemical_shifts must have length N"
        else:
            if N == 3:
                chemical_shift = torch.tensor([2.2, 4.8, -3.1], dtype=self.torch_dtype, device=self.device) - 3.03
            else:
                raise ValueError("chemical_shifts must be provided in cfg for N != 3")
        
        if "t2" in self.cfg:
            t2 = torch.tensor(self.cfg["t2"], dtype=self.torch_dtype, device=self.device) / 1000
            assert t2.shape[0] == N, "t2 must have length N"
        else:
            if N == 3:
                t2 = torch.tensor([12, 32, 61], dtype=self.torch_dtype, device=self.device) / 1000
            elif N == 2:
                t2 = torch.tensor([12, 32], dtype=self.torch_dtype, device=self.device) / 1000
        
        if "bssfp_signal" in self.cfg:
            bssfp_signal = torch.tensor(self.cfg["bssfp_signal"], dtype=self.torch_dtype, device=self.device)
            assert bssfp_signal.shape[0] == N, "bssfp_signal must have length N"
        else:
            if N == 3:
                bssfp_signal = torch.tensor([0.0403, 0.3332, 0.218], dtype=self.torch_dtype, device=self.device)
            elif N == 2:
                bssfp_signal = torch.tensor([0.0403, 0.3332], dtype=self.torch_dtype, device=self.device)
        
        # Simulation setup
        sw_spec = 4065
        field_fact = 61.45
        image_size = [self.target_size, self.target_size]
        
        # Time vectors
        t_spec = torch.linspace(0, (self.spec_len - 1)/sw_spec, self.spec_len, 
                              dtype=self.torch_dtype, device=self.device)  # (spec_len,)
        t_spec_4d = t_spec.view(1, 1, self.spec_len, 1, 1).expand(
            B, self.dce_number, self.spec_len, *image_size)  # (B, DCE, spec_len, H, W)
        
        # Initialize k-space
        final_kspace = torch.zeros((B, self.dce_number, self.spec_len, *image_size), 
                                  dtype=self.complex_dtype, device=self.device)
        
        # Simulation loop over N components
        for i_chem in range(chemical_com):
            density = self.chemical_density_map[:, i_chem]  # (B, DCE, H, W)
            t2_ = t2[i_chem]
            shift = chemical_shift[i_chem]
            decay = torch.exp(-t_spec_4d / t2_)  # (B, DCE, spec_len, H, W)
            phase = torch.exp(1j * 2 * torch.pi * shift * field_fact * t_spec_4d)
            temp_imag = density.unsqueeze(2) * decay * phase * bssfp_signal[i_chem]  # (B, DCE, spec_len, H, W)
            temp_rxy_acq = fft_xspace_to_kspace_3d_batch(
                fft_xspace_to_kspace_3d_batch(temp_imag, dim=-2), dim=-1)
            final_kspace += temp_rxy_acq
        
        # Post-processing
        k_field_spec = torch.fft.fftshift(torch.fft.fft(final_kspace, dim=2), dim=2)
        gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
        
        def process_complex_data(data, abs_flag, device):
            if abs_flag:
                processed = data.abs()
                processed = processed / processed.amax(dim=(-4, -3, -2, -1), keepdim=True)
            else:
                magnitude = torch.sqrt(data.real**2 + data.imag**2)
                norm_factor = magnitude.amax(dim=(-4, -3, -2, -1), keepdim=True)
                norm_factor = torch.where(norm_factor == 0, torch.tensor(1.0, device=device), norm_factor)
                real = data.real / norm_factor
                imag = data.imag / norm_factor
                processed = torch.stack([real, imag], dim=1)
            return processed
        
        complex_data = {"gt": gt}
        
        if 'no' in self.cfg.get("return_type", []):
            noisy_data = final_kspace.clone()
            noise = torch.randn_like(noisy_data) * self.cfg.get('wei_no', {}).get('noise_level') * torch.abs(noisy_data).max()
            noisy_data += noise
            k_field_spec = torch.fft.fftshift(torch.fft.fft(noisy_data, dim=2), dim=2)
            complex_data["no"] = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
        
        if 'wei' in self.cfg.get("return_type", []):
            average_masks, average_list_sum, average_list = update_averaging(self.target_size, self.target_size, average=self.cfg['average'])
            expanded_mask = average_masks.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device).to(self.torch_dtype)
            weighted_data = final_kspace * expanded_mask
            k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_data, dim=2), dim=2)
            complex_data["wei"] = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
            
            if 'wei_no' in self.cfg.get("return_type", []):
                weighted_noisy_data = weighted_data.clone()
                noise = torch.randn_like(weighted_noisy_data) * self.cfg.get('wei_no', {}).get('noise_level') * torch.abs(weighted_noisy_data).max()
                noise = noise * expanded_mask / expanded_mask.max()
                weighted_noisy_data += noise
                k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_noisy_data, dim=2), dim=2)
                complex_data["wei_no"] = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
        
        if 'standard_mean' in self.cfg.get("return_type", []):
            average_masks, average_list_sum, average_list = update_averaging(self.target_size, self.target_size, average=self.cfg['average'])
            average_sum = int(average_list_sum / self.target_size / self.target_size)
            # i have got k_field_spec
            for i in range(average_sum):
                noisy_data = final_kspace.clone()
                noise = torch.randn_like(noisy_data) * self.cfg.get('wei_no', {}).get('noise_level') * torch.abs(noisy_data).max()
                noisy_data += noise

        
        result = {key: process_complex_data(data, abs, gt.device) for key, data in complex_data.items()}
        return result