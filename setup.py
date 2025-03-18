from setuptools import setup, find_packages

setup(
    name="deforestation-analyser",
    version="1.0.0",
    description="A tool for analyzing deforestation and forest fires using satellite imagery",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/DeforestationAnalyser",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>1.5.8",
        "numpy",
        "pandas",
        "geopandas",
        "rasterio",
        "scikit-image",
        "pillow",
        "matplotlib",
        "streamlit",
        "huggingface_hub>=0.25.0",
        "opencv-python-headless",
        "tqdm",
        "albumentations>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "deforestation-analyser=run:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
) 