import setuptools
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="unitxt",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="IBM Research",
    author_email="elron.bandel@ibm.com",
    description="Load any mixture of text to text data in one line of code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibm/unitxt",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    package_data={"unitxt": ["catalog/*.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
