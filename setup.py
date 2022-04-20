from setuptools import find_packages, setup
from pathlib import Path

NAME = 'auralflow'

REQUIREMENTS = [
    'torch',
    'torchaudio'
    'librosa',
    'numpy',
    'matplotlib',
    'ipython',
    'pyyaml',
    'tabulate',
    'torchinfo',
    'tqdm',
]

with open(Path("README.md"), "r") as file:
    long_description = file.read()

setup(
    name="auralflow",
    version="1.0",
    description="A modular source separation training package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kianzohoury/auralflow",
    author="Kian Zohoury",
    author_email="kzohoury@berkeley.edu",
    classifiers=[
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    # Exclude the build files.
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    # extras_requires=EXTRAS,

    # entry_points={
    #     'console_scripts': [
    #         'auralflow = parse_sessions.__main__:main'
    #     ]
)

