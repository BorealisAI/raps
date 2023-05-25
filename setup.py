# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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
