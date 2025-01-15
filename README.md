source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main
cd /home/data1/musong/workspace/python/MRSpy
python3 -m pip install --upgrade build
python3 -m build
pip3 uninstall mrspy -y
pip3 install dist/mrspy-0.1.0-py3-none-any.whl