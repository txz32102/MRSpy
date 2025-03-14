import torch
from typing import Union
import numpy as np
from mrspy.util import load_mrs_mat, resize_image_batch, fft_xspace_to_kspace_3d_batch, fft_kspace_to_xspace_3d_batch
import os
from mrspy.util.noise import add_gaussian_noise
from mrspy.util.blur import update_averaging

class BatchSimulation:
    def __init__(self, dce_number: int = 26, spec_len: int = 64, target_size: int = 32, cfg: dict = {}):
        self.dce_number = dce_number
        self.spec_len = spec_len
        self.target_size = target_size
        self.cfg = cfg
        self.dtype = cfg.get("dtype", "float")  # 'float', 'double', or 'half'
        self.device = cfg.get('device', 'cpu')  # Default to CPU if not specified
        self._set_dtype()
        self.tackle_default_cfg()
        self.chemical_dce_curve_new = None
        self.chemical_density_map = None

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

    def tackle_default_cfg(self):
        if self.cfg.get("average") is None:
            self.cfg['average'] = 263

    @torch.no_grad()
    def load_default_curve(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "../data", "gray_dynamic_curve.mat")
        data_path = os.path.normpath(data_path)
        chemical_dce_curve = load_mrs_mat(data_path, output_type="tensor", 
                                          dtype=self.torch_dtype, device=self.device)
        
        column_indices = torch.arange(0, 52, 2)  # 0,2,4,...,50 (对应Matlab 1,3,5,...,51)
        
        water_dy = chemical_dce_curve[2, column_indices]  # Matlab第3行
        glu_dy = chemical_dce_curve[0, column_indices]   # Matlab第1行 
        lac_dy = chemical_dce_curve[1, column_indices]   # Matlab第2行
        
        self.chemical_dce_curve_new = torch.stack((water_dy, glu_dy), dim=0).to(self.device)

    @torch.no_grad()
    def load_curve(self, curve: Union[str, torch.Tensor]) -> None:
        if isinstance(curve, torch.Tensor):
            self.chemical_dce_curve_new = curve.to(dtype=self.torch_dtype, device=self.device)
        else:
            self.chemical_dce_curve_new = load_mrs_mat(curve, output_type="tensor", dtype=self.torch_dtype, device=self.device)

    @torch.no_grad()
    def generate_chemical_density_map(self, 
                                    water_image: Union[np.ndarray, "torch.Tensor"],
                                    glu_image: Union[np.ndarray, "torch.Tensor"]):
        water_image = torch.abs(water_image).unsqueeze(1).expand(-1, self.dce_number, *water_image.shape[-2:])  
        glu_image = torch.abs(glu_image).unsqueeze(1).expand(-1, self.dce_number, *glu_image.shape[-2:])

        W, H = self.target_size, self.target_size
        ChemicalDensityMap = torch.zeros(water_image.shape[0], 2, self.dce_number, W, H, dtype=self.torch_dtype, device=self.device)
        
        ChemicalDensityMap[:, 0] = water_image * self.chemical_dce_curve_new[0][:self.dce_number].view(1, -1, 1, 1)
        ChemicalDensityMap[:, 1] = glu_image * self.chemical_dce_curve_new[1][:self.dce_number].view(1, -1, 1, 1)
        
        self.chemical_density_map = ChemicalDensityMap

    @torch.no_grad()
    def simulation(self, water_img: Union[np.ndarray, "torch.Tensor"], 
                   glu_img: Union[np.ndarray, "torch.Tensor"],
                   abs=True):
        batch_size = water_img.shape[0]
        
        to_tensor = lambda x: torch.tensor(x, dtype=self.torch_dtype, device=self.device) if isinstance(x, np.ndarray) else x.to(self.device).to(self.torch_dtype)
        water_img, glu_img = map(to_tensor, [water_img, glu_img])
        water_img = resize_image_batch(water_img, self.target_size)
        glu_img = resize_image_batch(glu_img, self.target_size)

        curve = self.cfg.get("curve")
        if isinstance(curve, str):
            if curve == "default" or curve is None:
                self.load_default_curve()
        else:
            if isinstance(curve, str):
                self.load_curve(curve)
            else:
                self.chemical_dce_curve_new = torch.tensor(curve, dtype=torch.float32).to(self.device)

        self.generate_chemical_density_map(water_image=water_img, 
                                         glu_image=glu_img)

        chemical_com = 2
        if self.cfg.get("chemical_shifts") is None:
            chemical_shift = torch.tensor([2.2, 4.8], 
                                        dtype=self.torch_dtype, 
                                        device=self.device) - 3.03
        else:
            chemical_shift = torch.tensor(self.cfg.get("chemical_shifts"),
                                        dtype=self.torch_dtype, 
                                        device=self.device)[:chemical_com] - 3.03
            
        t2 = torch.tensor([12, 32], dtype=self.torch_dtype, device=self.device) / 1000
        field_fact = 61.45
        sw_spec = 4065
        image_size = [self.target_size, self.target_size]
        bssfp_signal = [0.0403, 0.3332]
        
        t_spec = torch.linspace(0, (self.spec_len - 1)/sw_spec, self.spec_len, 
                              dtype=self.torch_dtype, device=self.device)  
        t_spec_4d = t_spec.view(1, 1, self.spec_len, 1, 1).expand(
            batch_size, self.dce_number, self.spec_len, *image_size)  

        final_kspace = torch.zeros((batch_size, self.dce_number, self.spec_len, *image_size), 
                                  dtype=self.complex_dtype, device=self.device)

        for i_chem in range(chemical_com):
            density = self.chemical_density_map[:, i_chem]  
            density = density.unsqueeze(2)  
            t2_ = t2[i_chem]
            shift = chemical_shift[i_chem]
            
            decay = torch.exp(-t_spec_4d / t2_)  
            phase = torch.exp(1j * 2 * torch.pi * shift * field_fact * t_spec_4d)
            
            temp_imag = density * decay * phase * bssfp_signal[i_chem]  
            
            temp_rxy_acq = fft_xspace_to_kspace_3d_batch(
                fft_xspace_to_kspace_3d_batch(temp_imag, dim=-2),  
                dim=-1)  
            
            final_kspace += temp_rxy_acq

        k_field_spec = torch.fft.fftshift(torch.fft.fft(final_kspace, dim=2), dim=2)
        gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)

        if abs:
            gt = gt.abs() / gt.abs().amax(dim=(-4, -3, -2, -1), keepdim=True)
        else:
            gt = torch.stack([gt.real, gt.imag], dim=1)  

        result = {}
        if 'gt' in self.cfg.get("return_type"):
            result["gt"] = gt
            
        if 'no' in self.cfg.get("return_type"):
            noisy_data = final_kspace.clone()
            if abs:
                noisy_data = add_gaussian_noise(noisy_data, noise_level=self.cfg['wei_no']['noise_level'])
                k_field_spec = torch.fft.fftshift(torch.fft.fft(noisy_data, dim=2), dim=2)
                noisy_gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
                result["no"] = noisy_gt.abs() / noisy_gt.abs().amax(dim=(-4, -3, -2, -1), keepdim=True)
            else:
                noise_real = torch.randn_like(noisy_data.real) * self.cfg['wei_no']['noise_level']
                noise_imag = torch.randn_like(noisy_data.imag) * self.cfg['wei_no']['noise_level']
                noisy_data = noisy_data + torch.complex(noise_real, noise_imag)
                k_field_spec = torch.fft.fftshift(torch.fft.fft(noisy_data, dim=2), dim=2)
                noisy_gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
                result["no"] = torch.stack([noisy_gt.real, noisy_gt.imag], dim=1)  

        if 'wei' in self.cfg.get("return_type"):
            average_masks = update_averaging(self.target_size, self.target_size, average=self.cfg['average'])
            expanded_mask = average_masks.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
            weighted_data = final_kspace * expanded_mask.to(self.device).to(self.torch_dtype)
            k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_data, dim=2), dim=2)
            wei_gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
            if abs:
                result["wei"] = wei_gt.abs() / wei_gt.abs().amax(dim=(-4, -3, -2, -1), keepdim=True)
            else:
                result["wei"] = torch.stack([wei_gt.real, wei_gt.imag], dim=1)  
            
            if 'wei_no' in self.cfg.get("return_type"):
                weighted_noisy_data = weighted_data.clone()
                if abs:
                    weighted_noisy_data = add_gaussian_noise(weighted_noisy_data, noise_level=self.cfg['wei_no']['noise_level'])
                    k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_noisy_data, dim=2), dim=2)
                    wei_no_gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
                    result["wei_no"] = wei_no_gt.abs() / wei_no_gt.abs().amax(dim=(-4, -3, -2, -1), keepdim=True)
                else:
                    noise_real = torch.randn_like(weighted_noisy_data.real) * self.cfg['wei_no']['noise_level']
                    noise_imag = torch.randn_like(weighted_noisy_data.imag) * self.cfg['wei_no']['noise_level']
                    weighted_noisy_data = weighted_noisy_data + torch.complex(noise_real, noise_imag)
                    k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_noisy_data, dim=2), dim=2)
                    wei_no_gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
                    result["wei_no"] = torch.stack([wei_no_gt.real, wei_no_gt.imag], dim=1)  

        return result