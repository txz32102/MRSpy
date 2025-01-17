## dev mode

```bash
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main
cd /home/data1/musong/workspace/python/MRSpy
pip3 uninstall mrspy -y
pip install -e .
```

## deployment

```bash
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main
cd /home/data1/musong/workspace/python/MRSpy
pip3 uninstall mrspy -y
python3 setup.py sdist bdist_wheel
pip3 install dist/mrspy-0.1.0-py3-none-any.whl
python3 setup.py clean --all
```