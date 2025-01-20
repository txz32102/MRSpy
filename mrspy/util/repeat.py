import torch 

def repeat(img, dce_num):
    return img.repeat(dce_num, 1, 1)