from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent

setup(
    name="InterVelo",
    version="0.2.0",
    author="Yurou Wang",
    author_email="rapunzel@sjtu.edu.cn",
    description="A deep learning framework for joint inference of pseudotime and RNA velocity",
    long_description=(this_dir / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/yurouwang-rosie/InterVelo",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,

    install_requires=[
        # Core numerical stack
        "numpy>=1.23,<3.0",
        "scipy>=1.9,<2.0",
        "pandas>=1.5,<3.0",

        # ML / DL
        "torch>=2.0",
        "torchdiffeq>=0.2.3",

        # Single-cell ecosystem
        "scanpy>=1.9,<2.0",
        "scvelo>=0.3.1,<0.4",
        "anndata>=0.8,<0.11",

        # Performance / utilities
        "numba>=0.57",
        "scikit-learn>=1.2",
        "tqdm>=4.64",

        # Plotting
        "matplotlib>=3.7,<4.0",
        "seaborn>=0.12",
    ],

    python_requires=">=3.9,<3.12",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],

    entry_points={
        "console_scripts": [
            "intervelo=InterVelo.main:main", # 创建命令行工具
        ],
    },

    license="MIT",
)
