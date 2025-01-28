from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="vsa_ogm",
    version="0.0.1",
    install_requires=[
        "matplotlib",
        "numpy",
        "omegaconf",
        "opencv-python",
        "pandas",
        "pyntcloud",
        "pyyaml",
        "scikit-learn",
        "scikit-image",
        "tabulate",
        "tqdm",
        "wandb",
    ],
    author="Shay Snyder",
    author_email="ssnyde9@gmu.edu",
    description=("Occupancy Grid Mapping with Hyperdimensional Computing"),
    long_description=README,
    long_description_content_type="text/markdown",
    keywords="vector symbolic architectures, hyperdimensional computing",
    url="https://github.com/shaymeister/VSA-OGM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Progamming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires='>=3.9',
)