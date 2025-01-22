from setuptools import setup, find_packages

setup(
    name="mrspy",
    version="0.1.0",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.13.0",
        "matplotlib>=3.4.0",
        "h5py",
        "scipy"
    ],
    include_package_data=True, 
    package_data={
        "mrspy": ["data/*.mat"],
    },
)
