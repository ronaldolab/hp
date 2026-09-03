"""Packaging configuration for the local OpenMiChroM 1.1.1 source tree."""

from setuptools import find_packages, setup


setup(
    name="OpenMiChroM",
    version="1.1.1",
    description="Open-MiChroM library for chromosome simulations",
    url="https://ndb.rice.edu/Open-MiChroM",
    author="Antonio Bento de Oliveira Junior, Vinicius de Godoi Contessoto",
    author_email="antonio.oliveira@rice.edu, contessoto@rice.edu",
    packages=find_packages(),
    include_package_data=True,
    package_data={"OpenMiChroM": ["share/*.ff"]},
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "h5py",
        "pandas",
        "openmm",
    ],
    python_requires=">=3.9",
    zip_safe=False,
)
