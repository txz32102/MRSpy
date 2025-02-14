from mrspy.util.fast.datapipeline import datapipeline 

log_dir = "/home/data1/musong/workspace/2025/1/MRSpy/demo/fig"
data_dir_prefix = "/home/data1/musong/data/IXI/T1/fast/IXI216-HH-1635-T1/IXI216-HH-1635-T1_pve"
CSF_file_path = f"{data_dir_prefix}_0.nii.gz"
GM_file_path = f"{data_dir_prefix}_1.nii.gz"
WM_file_path = f"{data_dir_prefix}_2.nii.gz"

weight_dict={
    "CSF": [0.1, 0.3, 0.6],
    "GM": [0.3, 0.6, 0.2],
    "WM": [0.6, 0.1, 0],
}

size=(256, 256)
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

pipeline.plot(idx=157, cmap="hot", save_path=f"{log_dir}/water_glu_lac_plot.jpg", dpi=150)