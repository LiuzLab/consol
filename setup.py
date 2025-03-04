from setuptools import setup, find_packages

setup(
    name="consol",
    version="0.1.0",
    packages=find_packages('bin'),
    package_dir={'': 'bin'},
    install_requires=[],
    entry_points={
        'console_scripts': [
            'consol-cli=consol.main:main',
        ],
    },
    author="Jaeyeon Lee",
    author_email="Jaeyeon.Lee@bcm.edu",
    description="consol: Confident Solver to use LLM to solve various problems confidently and efficiently with a statistical approach.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LiuzLab/consol",
    python_requires=">=3.6",
)
