import os.path
from distutils.core import setup

setup(
    name="raps",
    version="0.1dev",
    description="Causal Bandit",
    author="Mikhail Konobeev",
    author_email="konobeev.michael@gmail.com",
    url="https://github.com/MichaelKonobeev/raps/",
    license="MIT",
    packages=["raps"],
    scripts=["raps/scripts/raps"],
    install_requires=[
        "jupyter==1.0.0",
        "matplotlib==3.7.1",
        "numpy==1.24.3",
        "threadpoolctl==3.1.0",
        "tqdm",
        f"u4ml @ file://localhost/{os.path.dirname(__file__)}/u4ml",
    ],
)
