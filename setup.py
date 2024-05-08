from setuptools import find_packages, setup

setup(
    name='sparkx',
    packages=find_packages(where='src/sparkx'),
    package_dir={'': 'sparkx'},
    install_requires=[
        "particle==0.23.0",
        "numpy>=1.23.5",
        "scipy>=1.10.1",
        "abc-property==1.0",
        "fastjet==3.4.1.3",
        "matplotlib>=3.7.1",
    ],
    version='1.2.1',
    description='Software Package for Analyzing Relativistic Kinematics in Collision eXperiments',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Niklas Götz, Hendrik Roch, Nils Sass',
    author_email="goetz@itp.uni-frankfurt.de, hroch@wayne.edu, nsass@itp.uni-frankfurt.de",
    url="https://smash-transport.github.io/sparkx/",
    download_url="https://github.com/smash-transport/sparkx",
    license='MIT',
    include_package_data = True
)
