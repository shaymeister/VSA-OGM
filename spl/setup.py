from setuptools import setup

if __name__ == "__main__":
    setup(
        name="spl",
        version="0.0.1",
        install_requires=[
            "ipykernel",
            "matplotlib",
            "numpy==1.26.0",
            "omegaconf",
            "scikit-learn",
            "tqdm",
        ],
        author="Shay Snyder",
        author_email="ssnyde9@gmu.edu",
        description = "Semantic Pointer Library",
        maintainer="Shay Snyder",
        maintainer_email="ssnyde9@gmu.edu",
        packages=["spl"]
    )