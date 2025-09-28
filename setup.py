from setuptools import setup, find_packages

setup(
    name="bardo-thodol-sim",
    version="0.1.0",
    description="Simulación cuántica de los estados post-mortem del Bardo Thödol",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'qiskit>=2.2.0',
        'numpy>=2.3.3',
        'matplotlib>=3.7.0',
        'scipy>=1.11.3',
        'seaborn>=0.12.2',
        'python-dotenv>=1.0.0',
    ],
    python_requires='>=3.10',
)
