import torch
from typing import Union
import numpy as np
from mrspy.util import load_mrs_mat, fft_xspace_to_kspace, fft_kspace_to_xspace, extract_center_kspace
import os
from mrspy.util.noise import add_gaussian_noise
from mrspy.util.blur import update_averaging

class Simulation:
    def __init__(self, dce_number: int = 32, spec_len: int = 120, target_size: int = 32, cfg: dict = {}):
        self.dce_number = dce_number
        self.spec_len = spec_len
        self.target_size = target_size
        self.chemical_dce_curve = None
        self.chemical_dce_curve_new = None
        self.chemical_density_map = None
        self.cfg = cfg
        self.dtype = cfg.get("dtype", "float")  # 'float', 'double', or 'half'
        self.device = cfg.get('device', 'cpu')  # Default to CPU if not specified
        self._set_dtype()

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

    @classmethod
    def from_mat():
        pass

    @classmethod
    def from_tensor(cls, tensor: Union[np.ndarray, "torch.Tensor"]):
        pass

    @torch.no_grad()
    def load_default_curve(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "../data", "gray_dynamic_curve.mat")
        data_path = os.path.normpath(data_path)
        self.chemical_dce_curve = load_mrs_mat(data_path, output_type="tensor", dtype=self.torch_dtype, device=self.device)

        # Vectorize operations for speed
        lac_dy = torch.cat([
            self.chemical_dce_curve[0, :20] * 2,
            self.chemical_dce_curve[0, 20:44:2] * 2
        ], dim=0)
        
        glu_dy = torch.cat([
            self.chemical_dce_curve[1, :16] * 4,
            self.chemical_dce_curve[1, 16:48:2] * 4
        ], dim=0)
        
        water_dy = torch.cat([
            self.chemical_dce_curve[2, :56:2],
            self.chemical_dce_curve[2, 56:60]
        ], dim=0)

        self.chemical_dce_curve_new = torch.stack((water_dy, glu_dy, lac_dy), dim=0).to(self.device)

    @torch.no_grad()
    def load_curve(self, path: str):
        self.chemical_dce_curve = load_mrs_mat(path, output_type="tensor", dtype=self.torch_dtype, device=self.device)
        # Use the same logic as in load_default_curve for processing
        self.load_default_curve()

    @torch.no_grad()
    def generate_chemical_density_map(self, 
                                water_image: Union[np.ndarray, "torch.Tensor"],
                                glu_image: Union[np.ndarray, "torch.Tensor"], 
                                lac_image: Union[np.ndarray, "torch.Tensor"]):
        water_image = torch.abs(water_image).unsqueeze(0).expand(self.dce_number, *water_image.shape)
        glu_image = torch.abs(glu_image).unsqueeze(0).expand(self.dce_number, *glu_image.shape)
        lac_image = torch.abs(lac_image).unsqueeze(0).expand(self.dce_number, *lac_image.shape)
        W, H = self.target_size, self.target_size
        ChemicalDensityMap = torch.zeros(3, self.dce_number, W, H)
        for i in range(self.dce_number):
            ChemicalDensityMap[0, i, :, :] = water_image[i, :, :] * self.chemical_dce_curve_new[2, i]
            ChemicalDensityMap[1, i, :, :] = glu_image[i, :, :] * self.chemical_dce_curve_new[0, i]
            ChemicalDensityMap[2, i, :, :] = lac_image[i, :, :] * self.chemical_dce_curve_new[1, i]
        self.chemical_density_map = ChemicalDensityMap

    @torch.no_grad()
    def simulation(self, water_img: Union[np.ndarray, "torch.Tensor"], glu_img: Union[np.ndarray, "torch.Tensor"], lac_img: Union[np.ndarray, "torch.Tensor"]):
        # Ensure inputs are in the right dtype and on the correct device
        to_tensor = lambda x: torch.tensor(x, dtype=self.torch_dtype, device=self.device) if isinstance(x, np.ndarray) else x.to(self.device).to(self.torch_dtype)

        water_img, glu_img, lac_img = map(to_tensor, [water_img, glu_img, lac_img])

        # FFT operations; assuming these functions are vectorized or optimized already and can handle device
        water_kspace = fft_xspace_to_kspace(water_img, 0)
        glu_kspace = fft_xspace_to_kspace(glu_img, 0)
        lac_kspace = fft_xspace_to_kspace(lac_img, 0)

        kspace_func = lambda img: extract_center_kspace(fft_xspace_to_kspace(img, 1), [self.target_size, self.target_size])
        water_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_func(water_kspace), 0), 1)
        glu_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_func(glu_kspace), 0), 1)
        lac_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_func(lac_kspace), 0), 1)

        if self.cfg.get("curve") is None or self.cfg.get("curve") == "default":
            self.load_default_curve()
        else:
            raise NotImplementedError("Custom curve functionality is not implemented yet.")

        self.generate_chemical_density_map(water_image=water_imag, glu_image=glu_imag, lac_image=lac_imag)
        self.chemical_density_map = self.chemical_density_map.to(self.device).to(self.torch_dtype)

        chemical_com = 3
        chemical_shift = torch.tensor([6.6, 3.8, -3.1, -13, 0.3], dtype=self.torch_dtype, device=self.device)[:chemical_com] - 3.03
        t2 = torch.tensor([12, 32, 61], dtype=self.torch_dtype, device=self.device) / 1000
        field_fact = 61.45
        sw_spec = 4065
        image_size = [self.target_size, self.target_size]
        
        t_spec = torch.linspace(0, (self.spec_len - 1) / sw_spec, self.spec_len, dtype=self.torch_dtype, device=self.device).unsqueeze(0)
        t_spec_3d = t_spec.view(self.spec_len, 1, 1).expand(self.spec_len, *image_size)

        final_kspace = torch.zeros((self.dce_number, self.spec_len, *image_size), dtype=self.complex_dtype, device=self.device)

        # Vectorize the loop for better performance
        for i_dce in range(self.dce_number):
            for i_chem in range(chemical_com):
                temp_imag = (
                    self.chemical_density_map[i_chem, i_dce].unsqueeze(0).expand_as(t_spec_3d)
                    * torch.exp(-t_spec_3d / t2[i_chem])
                    * torch.exp(1j * 2 * torch.pi * chemical_shift[i_chem] * field_fact * t_spec_3d)
                )
                temp_rxy_acq = fft_xspace_to_kspace(fft_xspace_to_kspace(temp_imag, 1), 2)
                final_kspace[i_dce] += temp_rxy_acq
        
        k_field_spec = torch.fft.fftshift(torch.fft.fft(final_kspace, dim=1), dim=1)
        gt = fft_kspace_to_xspace(fft_kspace_to_xspace(k_field_spec, 2), 3)
        gt = torch.abs(gt) / torch.max(torch.abs(gt))
        
        if(self.cfg == {}):
            return gt
        
        result = {}

        if 'gt' in self.cfg.get("return_type"):
            result["gt"] = gt
        
        if 'no' in self.cfg.get("return_type"):
            noisy_data = add_gaussian_noise(final_kspace,noise_level=0.02)
            k_field_spec = torch.fft.fftshift(torch.fft.fft(noisy_data, dim=1), dim=1)
            result["no"] = fft_kspace_to_xspace(fft_kspace_to_xspace(k_field_spec, 2), 3)
            result["no"] = torch.abs(result["no"]) / torch.max(torch.abs(result["no"]))
            
        if 'wei' in self.cfg.get("return_type"):
            average_masks = update_averaging(self.target_size, self.target_size)
            expanded_mask = average_masks.unsqueeze(0).unsqueeze(0)
            expanded_mask = expanded_mask.repeat(self.dce_number, self.spec_len, 1, 1)
            weighted_data = final_kspace * expanded_mask.to(self.device).to(self.torch_dtype)
            k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_data, dim=1), dim=1)
            result["wei"] = fft_kspace_to_xspace(fft_kspace_to_xspace(k_field_spec, 2), 3)
            result["wei"] = torch.abs(result["wei"]) / torch.max(torch.abs(result["wei"]))
            if 'wei_no' in self.cfg.get("return_type"):
                weighted_noisy_data = add_gaussian_noise(weighted_data,noise_level=0.02)
                k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_noisy_data, dim=1), dim=1)
                result["wei_no"] = fft_kspace_to_xspace(fft_kspace_to_xspace(k_field_spec, 2), 3)
                result["wei_no"] = torch.abs(result["wei_no"]) / torch.max(torch.abs(result["wei_no"]))

        return result