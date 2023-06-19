from setuptools import setup, find_packages

setup(
    name='unitxt',
    version='0.0.1',
    description='unitxt',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    author='Elron Bandel, IBM Research',
    author_email='elronbandel@ibm.com',
    url='https://github.com/IBM/unitxt',
)