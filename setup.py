from setuptools import setup, find_packages

setup(
    name='fr_models',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'torch',
        # 'torchdiffeq',
        'torch_interpolations @ git+https://www.github.com/sbarratt/torch_interpolations',
        'hyc-utils',
    ],
    extras_require={
        'test': ['pytest', 'pytest-benchmark'],
    }
)
