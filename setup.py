from setuptools import setup, find_packages

setup(
    name="consol",
    version="0.1.7",
    packages=find_packages('consol'),
    install_requires=[
        'python-dotenv',
        'scipy',
        'pandas',
        'langchain>=0.3.14',
        'langchain-openai>=0.3.3',
    ],
    entry_points={
        'console_scripts': [
            'consol=consol.main:main',
        ],
    },
    author=["Jaeyeon Lee", "Hyun-Hwan Jeong"],
    author_email=["Jaeyeon.Lee@bcm.edu", "Hyun-Hwan Jeong"],
    description="consol: Confident Solver to use LLM to solve various problems confidently and efficiently with a statistical approach.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LiuzLab/consol",
    python_requires=">=3.11.0",
)
