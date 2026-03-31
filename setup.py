from __future__ import annotations
from setuptools import setup, find_packages

setup(
    name="horseracing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "pdfplumber>=0.10.0",
        "pandas>=2.0.0",
    ],
)
