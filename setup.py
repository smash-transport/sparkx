from setuptools import find_packages, setup, Extension
import sys
import os

def build_extensions():
    ext_modules = []
    try:
        from Cython.Build import cythonize
        # Use the .pyx sources when Cython is available
        particle_sources = [os.path.join("src", "sparkx", "_particle_accel.pyx")]
        filter_sources = [os.path.join("src", "sparkx", "_filter_accel.pyx")]

        ext_particle = Extension(
            name="sparkx._particle_accel",
            sources=particle_sources,
            language="c",
        )
        ext_filter = Extension(
            name="sparkx._filter_accel",
            sources=filter_sources,
            language="c",
        )
        ext_modules = cythonize([ext_particle, ext_filter], compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        })
    except Exception:
        # No Cython at build time or cythonize failed. Fall back to no C-extension.
        # The pure-Python module src/sparkx/_particle_accel.py will be used.
        ext_modules = []
    return ext_modules

setup(
    name='sparkx',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "particle>=0.25.4",
        "numpy>=1.23.5",
        "scipy>=1.10.1",
        "fastjet>=3.4.2.1",
        "matplotlib>=3.7.1",
    ],
    version='2.1.1',
    description='Software Package for Analyzing Relativistic Kinematics in Collision eXperiments',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Nils Sass, Hendrik Roch, Niklas Götz, Renata Krupczak, Lucas Constantin',
    author_email="nsass@itp.uni-frankfurt.de, hroch@wayne.edu, goetz@itp.uni-frankfurt.de, rkrupczak@physik.uni-bielefeld.de, constantin@itp.uni-frankfurt.de",
    url="https://smash-transport.github.io/sparkx/",
    download_url="https://github.com/smash-transport/sparkx",
    license='GNU General Public License v3.0',
    include_package_data = True,
    ext_modules=build_extensions(),
)
