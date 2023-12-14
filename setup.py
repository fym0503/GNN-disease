import os

from setuptools import find_packages, setup

BUILD_ID = os.environ.get("BUILD_BUILDID", "0")

setup(
    name="gnn_disease",
    version="0.1" + "." + BUILD_ID,
    # Author details
    author="CSCI5120-Course-Project-AIH-Team",
    author_email="fanyimin@link.cuhk.edu.hk",
    packages=["gnn_disease"],
)
