import torch
from typing import Union
import numpy as np
from mrspy.util import load_mrs_mat
import os
from mrspy.util import *

class Simulation():
    def __init__(self, dce_number: int = 32, spec_len: int = 120, target_size: int = 32, cfg: dict = None):
        self.dce_number = dce_number
        self.spec_len = spec_len
        self.target_size = target_size
        self.chemical_dce_curve = None
        self.chemical_dce_curve_new = None
        self.chemical_density_map = None
        self.cfg = cfg

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
        self.chemical_dce_curve = load_mrs_mat(data_path, output_type="tensor")
        
        lac_dy_part1 = self.chemical_dce_curve[0, 0:20] * 2
        lac_dy_part2 = self.chemical_dce_curve[0, 20:44:2] * 2 
        lac_dy = torch.cat((lac_dy_part1, lac_dy_part2), dim=0)

        glu_dy_part1 = self.chemical_dce_curve[1, 0:16] * 4
        glu_dy_part2 = self.chemical_dce_curve[1, 16:48:2] * 4
        glu_dy = torch.cat((glu_dy_part1, glu_dy_part2), dim=0)

        water_dy_part1 = self.chemical_dce_curve[2, 0:56:2]
        water_dy_part2 = self.chemical_dce_curve[2, 56:60]
        water_dy = torch.cat((water_dy_part1, water_dy_part2), dim=0)

        self.chemical_dce_curve_new = torch.stack((water_dy, glu_dy, lac_dy), dim=0)

    @torch.no_grad()
    def load_curve(self, path: str):
        self.chemical_dce_curve = load_mrs_mat(path, output_type="tensor")
        lac_dy_part1 = self.chemical_dce_curve[0, 0:20] * 2
        lac_dy_part2 = self.chemical_dce_curve[0, 20:44:2] * 2 
        lac_dy = torch.cat((lac_dy_part1, lac_dy_part2), dim=0)

        glu_dy_part1 = self.chemical_dce_curve[1, 0:16] * 4
        glu_dy_part2 = self.chemical_dce_curve[1, 16:48:2] * 4
        glu_dy = torch.cat((glu_dy_part1, glu_dy_part2), dim=0)

        water_dy_part1 = self.chemical_dce_curve[2, 0:56:2]
        water_dy_part2 = self.chemical_dce_curve[2, 56:60]
        water_dy = torch.cat((water_dy_part1, water_dy_part2), dim=0)

        self.chemical_dce_curve_new = torch.stack((water_dy, glu_dy, lac_dy), dim=0)
        self.chemical_dce_curve_new = self.chemical_dce_curve_new
        
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
        water_kspace_data = fft_xspace_to_kspace(fft_xspace_to_kspace(water_img, 0), 1)
        glu_kspace_data = fft_xspace_to_kspace(fft_xspace_to_kspace(glu_img, 0), 1)
        lac_kspace_data = fft_xspace_to_kspace(fft_xspace_to_kspace(lac_img, 0), 1)
        
        kspace_data_center = extract_center_kspace(water_kspace_data, [self.target_size, self.target_size])
        water_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_data_center, 0), 1)

        kspace_data_center = extract_center_kspace(glu_kspace_data, [self.target_size, self.target_size])
        glu_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_data_center, 0), 1)

        kspace_data_center = extract_center_kspace(lac_kspace_data, [self.target_size, self.target_size])
        lac_imag = fft_kspace_to_xspace(fft_kspace_to_xspace(kspace_data_center, 0), 1)
        
        self.load_default_curve()
        self.generate_chemical_density_map(water_image=water_imag, glu_image=glu_imag, lac_image=lac_imag)
        
        chemical_com = 3
        chemical_shift = torch.tensor([6.6, 3.8, -3.1, -13, 0.3]) - 3.03
        t2 = torch.tensor([12, 32, 61]) / 1000
        field_fact = 61.45
        sw_spec = 4065
        image_size = [self.target_size, self.target_size]
        chemical_shift = chemical_shift[:chemical_com]
        t2 = t2[:chemical_com]
        np_spec = self.spec_len
        t_spec = torch.linspace(0, (np_spec - 1) / sw_spec, np_spec).unsqueeze(0)
        t_spec_3d = t_spec.view(np_spec, 1, 1).expand(np_spec, image_size[0], image_size[1])
        final_kspace = torch.zeros((self.dce_number, t_spec.shape[1], *image_size), dtype=torch.complex64)

        for i_dce in range(self.dce_number):
            for i_chem in range(chemical_com):
                temp_chemical_density_map = self.chemical_density_map[i_chem, i_dce, :, :]
                temp_chemical_density_map_3d = temp_chemical_density_map.unsqueeze(0).repeat(t_spec.shape[1], 1, 1)
                temp_imag = (
                    temp_chemical_density_map_3d
                    * torch.exp(-t_spec_3d / t2[i_chem])
                    * torch.exp(1j * 2 * torch.pi * chemical_shift[i_chem] * field_fact * t_spec_3d)
                )

                temp_rxy_acq = fft_xspace_to_kspace(fft_xspace_to_kspace(temp_imag, 1), 2)
                final_kspace[i_dce] += temp_rxy_acq

        return final_kspace

    @torch.no_grad()
    def generate_chemical_density_map_new(self, 
                                water_image: Union[np.ndarray, "torch.Tensor"],
                                glu_image: Union[np.ndarray, "torch.Tensor"], 
                                lac_image: Union[np.ndarray, "torch.Tensor"]):
        W, H = self.target_size, self.target_size
        ChemicalDensityMap = torch.zeros(3, self.dce_number, W, H)
        ChemicalDensityMap = ChemicalDensityMap
        for i in range(self.dce_number):
            ChemicalDensityMap[0, i, :, :] = water_image[i, :, :] * self.chemical_dce_curve_new[2, i]
            ChemicalDensityMap[1, i, :, :] = glu_image[i, :, :] * self.chemical_dce_curve_new[0, i]
            ChemicalDensityMap[2, i, :, :] = lac_image[i, :, :] * self.chemical_dce_curve_new[1, i]
        self.chemical_density_map = ChemicalDensityMap