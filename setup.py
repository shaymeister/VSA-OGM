from setuptools import setup

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
        "tqdm"
    ],
    author="Shay Snyder",
    author_email="ssnyde9@gmu.edu",
    description=("Occupancy Grid Mapping with Hyperdimensional Computing"),
    keywords="vector symbolic architectures, hyperdimensional computing",
    url="https://github.com/shaymeister/highfrost",
    packages=["vsa_ogm"],
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)