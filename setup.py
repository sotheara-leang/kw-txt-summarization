from setuptools import setup, find_packages

setup(
    name="txt-summarization",
    version="0.1",
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"])
)