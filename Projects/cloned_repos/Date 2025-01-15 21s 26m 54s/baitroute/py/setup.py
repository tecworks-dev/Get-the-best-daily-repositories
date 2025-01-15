from setuptools import setup, find_packages

setup(
    name="baitroute",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=5.1",
    ],
    author="Utku Sen",
    author_email="utkusen@gmail.com",
    description="A configurable HTTP honeypot library for Python web applications",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/utkusen/baitroute",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
) 