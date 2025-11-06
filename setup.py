# setup.py
from setuptools import setup, find_packages

setup(
    name="bardo-thodol-simulation",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "qutip>=4.7.0",
        "seaborn>=0.11.0",
    ],
    author="Horacio Héctor Hamann",
    author_email="via https://github.com/arathorian/BardoThodol",
    description="Quantum simulation of consciousness states based on Bardo Thödol",
    url="https://github.com/arathorian/BardoThodol",
    python_requires=">=3.8",
)