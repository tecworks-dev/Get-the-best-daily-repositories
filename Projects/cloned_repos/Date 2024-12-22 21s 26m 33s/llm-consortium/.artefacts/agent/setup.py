from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-consortium",
    version="0.1.0",
    author="ShellLM",
    author_email="info@shelllm.com",
    description="LLM Plugin for managing a consortium of language models with arbitration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shelllm/llm-consortium",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "llm>=0.5.0",
        "asyncio>=3.4.3",
        "typing>=3.7.4",
        "dataclasses;python_version<'3.7'",
    ],
    entry_points={
        "llm": ["consortium=consortium:ConsortiumPlugin"],
    },
)
