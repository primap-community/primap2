#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as history_file:
    history = history_file.read()

requirements = [
    "xarray",
    "pint",
    "pint_xarray",
    "numpy",
    "pandas",
    "openscm_units",
    "loguru",
    "scipy",
    "h5netcdf>=0.10",
    "h5py",
    "bottleneck",
    "matplotlib",
]

setup_requirements = [
    "pytest-runner",
]

setup(
    author="Mika Pflüger",
    author_email="mika.pflueger@pik-potsdam.de",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        #        "Programming Language :: Python :: 3.9",
    ],
    description="The next generation of the PRIMAP climate policy analysis suite.",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="primap2",
    name="primap2",
    packages=find_packages(include=["primap2", "primap2.*"]),
    setup_requires=setup_requirements,
    url="https://github.com/pik-primap/primap2",
    version="0.4.0",
    zip_safe=False,
)
