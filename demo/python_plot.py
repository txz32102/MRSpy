from mrspy.plot import SpecPlotter, plot_chemicalshift_image
from mrspy.util import load_mrs_mat

log_dir = "/home/data1/musong/workspace/2025/1/MRSpy/demo/fig"
data = load_mrs_mat("/home/data1/musong/workspace/2025/1/1-22/log/mat_data/wei_no.mat", output_type="tensor")
plotter = SpecPlotter.from_tensor(data)

plotter.spec_plot(path=f"{log_dir}/wei_no_spec_python.png", plot_all=True, dpi=600)

plot_chemicalshift_image(data, path=f"{log_dir}/python_wei_no", dpi=600)
