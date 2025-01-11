from setuptools import setup, find_namespace_packages
from restic_compose_backup import __version__

setup(
    name="restic-compose-backup",
    url="https://github.com/lawndoc/stack-back",
    version=__version__,
    author="Einar Forselv",
    author_email="eforselv@gmail.com",
    packages=find_namespace_packages(include=[
        'restic_compose_backup',
        'restic_compose_backup.*',
    ]),
    install_requires=[
        'docker~=7.1.0',
    ],
    entry_points={'console_scripts': [
        'restic-compose-backup = restic_compose_backup.cli:main',
        'rcb = restic_compose_backup.cli:main',
    ]},
)
