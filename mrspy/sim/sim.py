import torch
from typing import Optional, Union, List
import numpy as np
from mrspy.util import load_mrs_mat
import os

class ChemicalDensityMap():
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

    def load_default_curve(self):
        base_path = os.path.dirname(os.path.abspath(__file__)) 
        data_path = os.path.join(base_path, "../data", "gray_dynamic_curve.mat") 
        data_path = os.path.normpath(data_path) 
        self.chemical_dce_curve = load_mrs_mat(data_path, output_type="tensor")
        
        lac_dy_part1 = self.chemical_dce_curve[0, 0:20] * 2
        lac_dy_part2 = self.chemical_dce_curve[0, 20:43:2] * 2 
        lac_dy = torch.cat((lac_dy_part1, lac_dy_part2), dim=0)

        glu_dy_part1 = self.chemical_dce_curve[1, 0:16] * 4
        glu_dy_part2 = self.chemical_dce_curve[1, 16:48:2] * 4
        glu_dy = torch.cat((glu_dy_part1, glu_dy_part2), dim=0)

        water_dy_part1 = self.chemical_dce_curve[2, 1:56:2]
        water_dy_part2 = self.chemical_dce_curve[2, 56:60]
        water_dy = torch.cat((water_dy_part1, water_dy_part2), dim=0)

        self.chemical_dce_curve_new = torch.stack((water_dy, glu_dy, lac_dy), dim=0)

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
