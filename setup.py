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
        # 'torchdiffeq',
        # 'torch_interpolations @ git+https://www.github.com/sbarratt/torch_interpolations',
        'hyc-utils>=0.5.3',
    ],
    extras_require={
        'test': ['pytest', 'pytest-benchmark'],
    }
)
