from setuptools import setup, find_packages

setup(
    name='fr_models',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'torch',
        'torchdiffeq',
    ]
)