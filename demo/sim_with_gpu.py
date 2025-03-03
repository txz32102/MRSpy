from mrspy.util import load_mrs_mat
from mrspy.sim import Simulation
import os
import time
from mrspy.plot import SpecPlotter, plot_chemicalshift_image
from mrspy.util import load_mrs_mat

demo_folder = "/home/data1/data/dmi_si_hum/data_metimg/row0_IXI255-HH-1882-T1"
log_dir = "/home/data1/musong/workspace/2025/3/03-03/log/noise_plot"

water_img = os.path.join(demo_folder, "WaterImag.mat")
water_img = load_mrs_mat(water_img, output_type="tensor")
water_img = water_img.double()

glu_img = os.path.join(demo_folder, "GluImag.mat")
glu_img = load_mrs_mat(glu_img, output_type="tensor")
glu_img = glu_img.double()

lac_img = os.path.join(demo_folder, "LacImag.mat")
lac_img = load_mrs_mat(lac_img, output_type="tensor")
lac_img = lac_img.double()

cfg = {
    "curve": "default",
    "device": "cuda:0",
    "return_type": {
        "gt",
        "no",
        "wei",
        "wei_no"},
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

start_time = time.time()
target_size = 32
sim = Simulation(target_size=target_size, cfg=cfg)

# we will have res["gt"], res["no"], res["wei"], res["wei_no], each of shape 32, 120, 32, 32
res = sim.simulation(water_img=water_img, glu_img=glu_img, lac_img=lac_img)
end_time = time.time()
print(f"Total time taken by GPU: {end_time - start_time:.2f} seconds")

# plot the weighted average + noise image
res['wei_no'] = res['wei_no'].cpu()
plotter = SpecPlotter.from_tensor(res['wei_no'])

plotter.spec_plot(path=f"{log_dir}/wei_no_spec_python.png", plot_all=True, dpi=100, show_xy=True)

plot_chemicalshift_image(res['wei_no'], path=f"{log_dir}/python_wei_no", dpi=100, normalize_range=True)