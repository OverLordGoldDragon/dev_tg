import os
import re
from setuptools import setup, find_packages

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return reader.read()


def get_requirements(*parts):
    with open(os.path.join(current_path, *parts), encoding='utf-8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_matched = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                version_file, re.M)
    if version_matched:
        return version_matched.group(1)
    raise RuntimeError('Unable to find version')


setup(
    name="deeptrain",
    version=find_version('deeptrain', '__init__.py'),
    packages=find_packages(exclude=['tests', 'examples']),
    url="https://github.com/OverLordGoldDragon/dev_tg",
    license="MIT",
    author="OverLordGoldDragon",
    author_email="16495490+OverLordGoldDragon@users.noreply.github.com",
    description=("dev-stage repo"),
    long_description=read_file('README.md'),
    long_description_content_type="text/markdown",
    keywords=(
        "tensorflow keras deep-learning"
    ),
    install_requires=get_requirements('requirements.txt'),
    extras_require={
        "docs": get_requirements('requirements-dev.txt'),
        "travis": get_requirements('requirements-dev.txt'),
        },
    tests_require=["pytest>=4.0", "pytest-cov"],
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)