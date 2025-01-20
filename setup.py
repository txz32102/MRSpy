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
        # Add dependencies here, e.g., "numpy", "matplotlib"
    ],
    include_package_data=True, 
    package_data={
        "mrspy": ["data/*.mat"],  # Specify the relative path to your data
    },
)
