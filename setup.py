import setuptools

setuptools.setup(
    name="Auralate",
    packages=["Auralate"],
    entry_points={
        'console_scripts': [
            'Auralate = parse_sessions.__main__:main'
        ]
    }
)
