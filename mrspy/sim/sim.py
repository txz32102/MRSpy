import torch
from typing import Optional, Union, List
import numpy as np
from mrspy.util import load_mrs_mat
import os
from mrspy.util import *

class Simulation():
    def __init__(self, dce_number: int = 32, spec_len: int = 120, image_size: int = 32):
        self.dce_number = dce_number
        self.spec_len = spec_len
        self.image_size = image_size
        self.chemical_dce_curve = None
        self.chemical_dce_curve_new = None
        self.chemical_density_map = None

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
        self.chemical_dce_curve_new = self.chemical_dce_curve_new.double()
        
    @torch.no_grad()
    def generate_chemical_density_map(self, 
                                water_image: Union[np.ndarray, "torch.Tensor"],
                                glu_image: Union[np.ndarray, "torch.Tensor"], 
                                lac_image: Union[np.ndarray, "torch.Tensor"]):
        water_image = water_image.repeat(self.dce_number, 1, 1)
        glu_image = glu_image.repeat(self.dce_number, 1, 1)
        lac_image = lac_image.repeat(self.dce_number, 1, 1)
        W, H = self.image_size, self.image_size
        ChemicalDensityMap = torch.zeros(3, self.dce_number, W, H)
        for i in range(self.dce_number):
            ChemicalDensityMap[0, i, :, :] = water_image[i, :, :] * self.chemical_dce_curve_new[0, i]
            ChemicalDensityMap[1, i, :, :] = glu_image[i, :, :] * self.chemical_dce_curve_new[1, i]
            ChemicalDensityMap[2, i, :, :] = lac_image[i, :, :] * self.chemical_dce_curve_new[2, i]
        self.chemical_density_map = ChemicalDensityMap

    @torch.no_grad()
    def simulation(self, field_fact: float = 61.45, sw_spec: int = 4065):
        chemical_com = 3
        chemical_shift = torch.tensor([6.6, 3.8, -3.1, -13, 0.3]) - 3.03
        t2 = torch.tensor([12, 32, 61]) / 1000

        image_size = [self.image_size, self.image_size]
        chemical_shift = chemical_shift[:chemical_com]
        t2 = t2[:chemical_com]
        np_spec = self.spec_len
        t_spec = torch.arange(0, np_spec) / sw_spec
        t_spec_3d = t_spec.view(-1, 1, 1).repeat(1, image_size[0], image_size[1])

        final_kspace = torch.zeros((self.dce_number, len(t_spec), *image_size), dtype=torch.complex64)

        for i_dce in range(self.dce_number):
            for i_chem in range(chemical_com):
                temp_chemical_density_map = self.chemical_density_map[i_chem, i_dce, :, :]
                temp_chemical_density_map_3d = temp_chemical_density_map.unsqueeze(0).repeat(len(t_spec), 1, 1)
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
        W, H = self.image_size, self.image_size
        ChemicalDensityMap = torch.zeros(3, self.dce_number, W, H)
        ChemicalDensityMap = ChemicalDensityMap.double()
        for i in range(self.dce_number):
            ChemicalDensityMap[0, i, :, :] = water_image[i, :, :] * self.chemical_dce_curve_new[2, i]
            ChemicalDensityMap[1, i, :, :] = glu_image[i, :, :] * self.chemical_dce_curve_new[0, i]
            ChemicalDensityMap[2, i, :, :] = lac_image[i, :, :] * self.chemical_dce_curve_new[1, i]
        self.chemical_density_map = ChemicalDensityMap