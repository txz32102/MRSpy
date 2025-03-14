import torch
from typing import Union
import numpy as np
from mrspy.util import load_mrs_mat, fft_xspace_to_kspace, fft_kspace_to_xspace, resize_image, resize_image_batch, fft_xspace_to_kspace_3d_batch, fft_kspace_to_xspace_3d_batch
import os
from mrspy.util.noise import add_gaussian_noise
from mrspy.util.blur import update_averaging

class Simulation:
    def __init__(self, dce_number: int = 26, spec_len: int = 64, target_size: int = 32, cfg: dict = {}):
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
        # 数据加载保持不变
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, "../data", "gray_dynamic_curve.mat")
        data_path = os.path.normpath(data_path)
        self.chemical_dce_curve = load_mrs_mat(data_path, output_type="tensor", 
                                            dtype=self.torch_dtype, device=self.device)
        
        # 生成Matlab对应的列索引 (2i-1的Python等效)
        column_indices = torch.arange(0, 52, 2)  # 0,2,4,...,50 (对应Matlab 1,3,5,...,51)
        
        # 按Matlab索引规则选择数据（去除缩放因子）
        water_dy = self.chemical_dce_curve[2, column_indices]  # Matlab第3行
        glu_dy = self.chemical_dce_curve[0, column_indices]   # Matlab第1行 
        lac_dy = self.chemical_dce_curve[1, column_indices]   # Matlab第2行
        
        # 堆叠维度顺序与Matlab一致
        self.chemical_dce_curve_new = torch.stack((water_dy, glu_dy, lac_dy), dim=0).to(self.device)

    @torch.no_grad()
    def load_curve(self, curve: Union[str, torch.Tensor]) -> None:
        if isinstance(curve, torch.Tensor):
            self.chemical_dce_curve = curve.to(dtype=self.torch_dtype, device=self.device)
        else:
            self.chemical_dce_curve = load_mrs_mat(curve, output_type="tensor", dtype=self.torch_dtype, device=self.device)
        
        self.chemical_dce_curve_new = self.chemical_dce_curve

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
            ChemicalDensityMap[0, i, :, :] = water_image[i, :, :] * self.chemical_dce_curve_new[0, i]
            ChemicalDensityMap[1, i, :, :] = glu_image[i, :, :] * self.chemical_dce_curve_new[1, i]
            ChemicalDensityMap[2, i, :, :] = lac_image[i, :, :] * self.chemical_dce_curve_new[2, i]
        self.chemical_density_map = ChemicalDensityMap

    @torch.no_grad()
    def simulation(self, water_img: Union[np.ndarray, "torch.Tensor"], glu_img: Union[np.ndarray, "torch.Tensor"], lac_img: Union[np.ndarray, "torch.Tensor"]):
        # Ensure inputs are in the right dtype and on the correct device
        to_tensor = lambda x: torch.tensor(x, dtype=self.torch_dtype, device=self.device) if isinstance(x, np.ndarray) else x.to(self.device).to(self.torch_dtype)

        water_img, glu_img, lac_img = map(to_tensor, [water_img, glu_img, lac_img])

        """
        this is the old resize method that was used to do the resize in the kspace, we can simply did this in image domain, no need for kspace 

        # FFT operations; assuming these functions are vectorized or optimized already and can handle device
        water_kspace = fft_xspace_to_kspace(water_img, 0)
        glu_kspace = fft_xspace_to_kspace(glu_img, 0)
        lac_kspace = fft_xspace_to_kspace(lac_img, 0)

        kspace_func = lambda img: extract_center_kspace(fft_xspace_to_kspace(img, 1), [self.target_size, self.target_size])
        water_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_func(water_kspace), 0), 1)
        glu_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_func(glu_kspace), 0), 1)
        lac_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_func(lac_kspace), 0), 1)
        """
        water_img = resize_image(water_img, self.target_size)
        glu_img = resize_image(glu_img, self.target_size)
        lac_img = resize_image(lac_img, self.target_size)

        water_kspace = fft_xspace_to_kspace(fft_xspace_to_kspace(water_img, 0), 1)
        glu_kspace = fft_xspace_to_kspace(fft_xspace_to_kspace(glu_img, 0), 1)
        lac_kspace = fft_xspace_to_kspace(fft_xspace_to_kspace(lac_img, 0), 1)

        water_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(water_kspace, 0), 1)
        glu_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(glu_kspace, 0), 1)
        lac_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(lac_kspace, 0), 1)

        if self.cfg.get("curve") is None or self.cfg.get("curve") == "default":
            self.load_default_curve()
        else:
            self.load_curve(self.cfg.get("curve"))

        self.generate_chemical_density_map(water_image=water_imag, glu_image=glu_imag, lac_image=lac_imag)
        self.chemical_density_map = self.chemical_density_map.to(self.device).to(self.torch_dtype)

        chemical_com = 2
        chemical_shift = torch.tensor([4.8, 2.2, 1.3, -13, 0.3], dtype=self.torch_dtype, device=self.device)[:chemical_com] - 3.03
        t2 = torch.tensor([12, 32, 61], dtype=self.torch_dtype, device=self.device) / 1000
        field_fact = 61.45
        sw_spec = 4065
        image_size = [self.target_size, self.target_size]
        bssfp_signal = [0.0403, 0.3332, 0.218]
        
        t_spec = torch.linspace(0, (self.spec_len - 1) / sw_spec, self.spec_len, dtype=self.torch_dtype, device=self.device).unsqueeze(0)
        t_spec_3d = t_spec.view(self.spec_len, 1, 1).expand(self.spec_len, *image_size)

        final_kspace = torch.zeros((self.dce_number, self.spec_len, *image_size), dtype=self.complex_dtype, device=self.device)

        # Vectorize the loop for better performance
        for i_dce in range(self.dce_number):
            for i_chem in range(chemical_com):
                temp_imag = (
                    bssfp_signal[i_chem] * self.chemical_density_map[i_chem, i_dce].unsqueeze(0).expand_as(t_spec_3d)
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
            noisy_data = add_gaussian_noise(final_kspace,noise_level=self.cfg['wei_no']['noise_level'])
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
                weighted_noisy_data = add_gaussian_noise(weighted_data,noise_level=self.cfg['wei_no']['noise_level'])
                k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_noisy_data, dim=1), dim=1)
                result["wei_no"] = fft_kspace_to_xspace(fft_kspace_to_xspace(k_field_spec, 2), 3)
                result["wei_no"] = torch.abs(result["wei_no"]) / torch.max(torch.abs(result["wei_no"]))

        return result

class BatchSimulation(Simulation):
    def __init__(self, dce_number: int = 26, spec_len: int = 64, target_size: int = 32, cfg: dict = {}):
        super().__init__(dce_number, spec_len, target_size, cfg)
        self.tackle_default_cfg()
        
    def tackle_default_cfg(self):
        if(self.cfg.get("average") is None):
            self.cfg['average'] = 263
        
    @torch.no_grad()
    def generate_chemical_density_map(self, 
                                water_image: Union[np.ndarray, "torch.Tensor"],
                                glu_image: Union[np.ndarray, "torch.Tensor"], 
                                lac_image: Union[np.ndarray, "torch.Tensor"]):
        # Input shapes: (B, H, W)
        water_image = torch.abs(water_image).unsqueeze(1).expand(-1, self.dce_number, *water_image.shape[-2:])  # (B, DCE, H, W)
        glu_image = torch.abs(glu_image).unsqueeze(1).expand(-1, self.dce_number, *glu_image.shape[-2:])
        lac_image = torch.abs(lac_image).unsqueeze(1).expand(-1, self.dce_number, *lac_image.shape[-2:])

        # Expand curves to match batch dimensions
        W, H = self.target_size, self.target_size
        ChemicalDensityMap = torch.zeros(water_image.shape[0], 3, self.dce_number, W, H, dtype=self.torch_dtype, device=self.device)
        
        # Vectorized computation for all batches and timepoints
        ChemicalDensityMap[:, 0] = water_image * self.chemical_dce_curve_new[0][:self.dce_number].view(1, -1, 1, 1)
        ChemicalDensityMap[:, 1] = glu_image * self.chemical_dce_curve_new[1][:self.dce_number].view(1, -1, 1, 1)
        ChemicalDensityMap[:, 2] = lac_image * self.chemical_dce_curve_new[2][:self.dce_number].view(1, -1, 1, 1)
        
        self.chemical_density_map = ChemicalDensityMap

    @torch.no_grad()
    def simulation(self, water_img: Union[np.ndarray, "torch.Tensor"], 
                   glu_img: Union[np.ndarray, "torch.Tensor"], 
                   lac_img: Union[np.ndarray, "torch.Tensor"],
                   abs=True):
        # Input shapes: (B, H, W)
        batch_size = water_img.shape[0]
        
        # Convert inputs to tensor and move to device
        to_tensor = lambda x: torch.tensor(x, dtype=self.torch_dtype, device=self.device) if isinstance(x, np.ndarray) else x.to(self.device).to(self.torch_dtype)
        water_img, glu_img, lac_img = map(to_tensor, [water_img, glu_img, lac_img])
        water_img = resize_image_batch(water_img, self.target_size)
        glu_img = resize_image_batch(glu_img, self.target_size)
        lac_img = resize_image_batch(lac_img, self.target_size)

        # Load curves (same as parent class)
        curve = self.cfg.get("curve")
        if isinstance(curve, str):
            if(curve == "default" or curve is None):
                self.load_default_curve()
        else:
            if isinstance(curve, str):
                self.load_curve(curve)
            else:
                self.chemical_dce_curve_new = torch.tensor(curve, dtype=torch.float32).to(self.device)

        # Generate density maps with batch support
        self.generate_chemical_density_map(water_image=water_img, 
                                         glu_image=glu_img, 
                                         lac_image=lac_img)

        # Simulation parameters
        chemical_com = 2
        if(self.cfg.get("chemical_shifts") is None):
            chemical_shift = torch.tensor([2.2, 4.8, -3.1, -13, 0.3], 
                                        dtype=self.torch_dtype, 
                                        device=self.device)[:chemical_com] - 3.03
        else:
            chemical_shift = torch.tensor(self.cfg.get("chemical_shifts"),
                                        dtype=self.torch_dtype, 
                                        device=self.device)[:chemical_com] - 3.03
            
        t2 = torch.tensor([12, 32, 61], dtype=self.torch_dtype, device=self.device) / 1000
        field_fact = 61.45
        sw_spec = 4065
        image_size = [self.target_size, self.target_size]
        bssfp_signal = [0.0403, 0.3332, 0.218]
        
        # Time vectors with batch and DCE dimensions
        t_spec = torch.linspace(0, (self.spec_len - 1)/sw_spec, self.spec_len, 
                              dtype=self.torch_dtype, device=self.device)  # (spec_len,)
        t_spec_4d = t_spec.view(1, 1, self.spec_len, 1, 1).expand(
            batch_size, self.dce_number, self.spec_len, *image_size)  # (B, DCE, spec_len, H, W)

        # Initialize final kspace with batch dimension
        final_kspace = torch.zeros((batch_size, self.dce_number, self.spec_len, *image_size), 
                                  dtype=self.complex_dtype, device=self.device)

        # Vectorized computation across chemicals and DCE timepoints
        for i_chem in range(chemical_com):
            # Get density maps for current chemical (B, DCE, H, W)
            density = self.chemical_density_map[:, i_chem]  # (B, DCE, H, W)
            
            # Expand dimensions for broadcasting
            density = density.unsqueeze(2)  # (B, DCE, 1, H, W)
            t2_ = t2[i_chem]
            shift = chemical_shift[i_chem]
            
            # Compute exponential terms
            decay = torch.exp(-t_spec_4d / t2_)  # (B, DCE, spec_len, H, W)
            phase = torch.exp(1j * 2 * torch.pi * shift * field_fact * t_spec_4d)
            
            # Combine terms and compute FFTs
            temp_imag = density * decay * phase * bssfp_signal[i_chem]  # (B, DCE, spec_len, H, W)
            
            # Apply 2D FFT for each timepoint
            temp_rxy_acq = fft_xspace_to_kspace_3d_batch(
                fft_xspace_to_kspace_3d_batch(temp_imag, dim=-2),  # Spatial dim 1
                dim=-1)  # Spatial dim 2
            
            # Accumulate results
            final_kspace += temp_rxy_acq

        # Spectral FFT and processing
        k_field_spec = torch.fft.fftshift(torch.fft.fft(final_kspace, dim=2), dim=2)
        gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)

        # Helper function to process complex data based on the abs flag
        def process_complex_data(data, abs_flag, device):
            if abs_flag:
                # Take absolute value and normalize
                processed = data.abs()
                processed = processed / processed.amax(dim=(-4, -3, -2, -1), keepdim=True)
            else:
                # Split into real and imaginary parts
                real = data.real
                imag = data.imag
                # Compute normalization factor from real part's max absolute value
                norm_factor = real.abs().amax(dim=(-4, -3, -2, -1), keepdim=True)
                norm_factor = torch.where(norm_factor == 0, torch.tensor(1.0, device=device), norm_factor)
                # Normalize both parts
                real = real / norm_factor
                imag = imag / norm_factor
                # Stack them together (B, 2, T, L, W, H)
                processed = torch.stack([real, imag], dim=1)
            return processed

        # Main processing logic
        complex_data = {}

        # Compute ground truth (assuming gt is already computed as complex data)
        complex_data["gt"] = gt

        # Compute noisy data if required
        if 'no' in self.cfg.get("return_type"):
            noisy_data = final_kspace.clone()
            # Add complex noise (consistent for both abs and else cases)
            noise = torch.randn_like(noisy_data) * self.cfg['wei_no']['noise_level']
            noisy_data = noisy_data + noise
            k_field_spec = torch.fft.fftshift(torch.fft.fft(noisy_data, dim=2), dim=2)
            noisy_gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
            complex_data["no"] = noisy_gt

        # Compute weighted and weighted noisy data if required
        if 'wei' in self.cfg.get("return_type"):
            average_masks = update_averaging(self.target_size, self.target_size, average=self.cfg['average'])
            expanded_mask = average_masks.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device).to(self.torch_dtype)
            weighted_data = final_kspace * expanded_mask
            k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_data, dim=2), dim=2)
            wei_gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
            complex_data["wei"] = wei_gt

            if 'wei_no' in self.cfg.get("return_type"):
                weighted_noisy_data = weighted_data.clone()
                # Add complex noise
                noise = torch.randn_like(weighted_noisy_data) * self.cfg['wei_no']['noise_level']
                weighted_noisy_data = weighted_noisy_data + noise
                k_field_spec = torch.fft.fftshift(torch.fft.fft(weighted_noisy_data, dim=2), dim=2)
                wei_no_gt = fft_kspace_to_xspace_3d_batch(fft_kspace_to_xspace_3d_batch(k_field_spec, dim=-1), dim=-2)
                complex_data["wei_no"] = wei_no_gt

        # Process all complex data and store results
        result = {}
        for key in complex_data:
            result[key] = process_complex_data(complex_data[key], abs, gt.device)

        return result
