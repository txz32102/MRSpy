from setuptools import setup, find_packages

setup(
    name="mrspy",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A personal Python library for simulation and plotting tools.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mrspy",  # Replace with your repo URL if hosted online
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
)
