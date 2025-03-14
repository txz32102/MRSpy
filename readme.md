## A python library use torch intended to make the MRS simulation faster

Basic usage of GPU

```python
from mrspy.plot import SpecPlotter, plot_chemicalshift_image
from mrspy.sim.sim import Simulation
from mrspy.util import load_mrs_mat

demo_folder = "/home/data1/data/dmi_si_hum/data_metimg/row0_IXI255-HH-1882-T1"

water_img = os.path.join(demo_folder, "WaterImag.mat")
water_img = load_mrs_mat(water_img, output_type="tensor")
water_img = water_img.double()

glu_img = os.path.join(demo_folder, "GluImag.mat")
glu_img = load_mrs_mat(glu_img, output_type="tensor")
glu_img = glu_img.double()

lac_img = os.path.join(demo_folder, "LacImag.mat")
lac_img = load_mrs_mat(lac_img, output_type="tensor")
lac_img = lac_img.double()

torch.stack([water_img.unsqueeze(0), glu_img.unsqueeze(0), lac_img.unsqueeze(0)], dim=1).shape

sim = Simulation()
res = sim.simulation(torch.stack([water_img.unsqueeze(0), glu_img.unsqueeze(0), lac_img.unsqueeze(0)], dim=1))

log_dir = "temp"

plot_chemicalshift_image(res['gt'].cpu()[0], chemicalshift=[31, 33], path=f'{log_dir}/test_gt')
SpecPlotter.from_tensor(res['gt'].cpu()[0],).spec_plot(path=f"{log_dir}/gt_spec_python.png", plot_all=True, dpi=100)
plot_chemicalshift_image(res['wei_no'].cpu()[0], chemicalshift=[31, 33], path=f'{log_dir}/test_wei_no')
SpecPlotter.from_tensor(res['wei_no'].cpu()[0],).spec_plot(path=f"{log_dir}/wei_no_spec_python.png", plot_all=True, dpi=100)
```

## datapipeline usage

This script generates simulated water, glucose, and lactate images from CSF, GM, and WM tissue maps, using a custom weight dictionary. It processes and resizes the images, then saves them as PNG files.

```python
from mrspy.util.fast.datapipeline import datapipeline 
from mrspy.plot import plot

CSF_file_path = "/home/data1/musong/data/IXI/T1/fast/IXI216-HH-1635-T1/IXI216-HH-1635-T1_pve_0.nii.gz"
GM_file_path = "/home/data1/musong/data/IXI/T1/fast/IXI216-HH-1635-T1/IXI216-HH-1635-T1_pve_1.nii.gz"
WM_file_path = "/home/data1/musong/data/IXI/T1/fast/IXI216-HH-1635-T1/IXI216-HH-1635-T1_pve_2.nii.gz"

weight_dict={
    "CSF": [0.1, 0.3, 0.6],
    "GM": [0.3, 0.6, 0.2],
    "WM": [0.6, 0.1, 0],
}

size=(128, 128)
pipeline = datapipeline(CSF_file_path=CSF_file_path, 
                        GM_file_path=GM_file_path, 
                        WM_file_path=WM_file_path, 
                        weight_dict=weight_dict, 
                        size=size, 
                        pad=0.3, 
                        pad_axis="x")

pipeline.process()
water = pipeline.water
print(water.shape)
pipeline.plot(idx=157, cmap="hot")

pipeline.save(output_path="/home/data1/musong/workspace/2025/2/2-13/temp", output_type="png")
```

## install

```bash
pip install git+https://github.com/txz32102/MRSpy.git@0.3.3
```

Visualization results for `simulated.mat` can be viewed using MATLAB, available in the `demo` folder.

![Glutamate Ground Truth](./demo/fig/glu_gt.png)

![Water Ground Truth](./demo/fig/water_gt.png)

![Lactate Ground Truth](./demo/fig/lac_gt.png)

![Spectroscopy Ground Truth](./demo/fig/spectroscopy_plot_gt.png)
