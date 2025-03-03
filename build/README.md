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
pip3 install dist/mrspy-0.3.2-py3-none-any.whl
python3 setup.py clean --all
```

## sys path

```python
! pip3 uninstall mrspy -y
import sys
sys.path.append("/home/data1/musong/workspace/2025/1/MRSpy")
```