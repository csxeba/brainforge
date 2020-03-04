from setuptools import setup, find_packages

long_description = open("Readme.md").read()

setup(
    name='brainforge',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/csxeba/brainforge.git',
    license='GPLv3',
    author='csxeba',
    author_email='csxeba@gmail.com',
    description='Deep Learning with NumPy only!',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
