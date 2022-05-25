import os

from setuptools import find_packages, setup

src_dir = os.path.abspath(os.path.dirname(__file__))
# requirements_txt = os.path.join(src_dir, "requirements.txt")
with open("requirements.txt", encoding="utf8") as f:
    required = f.read().splitlines()


setup(
    name="lightningmnist",
    version="0.1",
    author="Davide",
    url="https://pytorch.org",
    install_requires=required,
    packages=find_packages(),
    include_package_data=True,
)
