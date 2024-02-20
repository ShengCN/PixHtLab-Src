#!/usr/bin/env python

import os, sys
from distutils.core import setup

setup_args = {}

#############################################################################################
##### CEBALERT: copied from topographica; should be simplified

required = {'param':">=1.3.1",
            'numpy':">=1.0",
            'holoviews':">=1.0.1"}

# could add tkinter, paramtk
# optional = {}

packages_to_install = [required]
packages_to_state = [required]

setup_args = {}

if 'setuptools' in sys.modules:
    # support easy_install without depending on setuptools
    install_requires = []
    for package_list in packages_to_install:
        install_requires+=["%s%s"%(package,version) for package,version in package_list.items()]
    setup_args['install_requires']=install_requires
    setup_args['dependency_links']=["http://pypi.python.org/simple/"]
    setup_args['zip_safe']=False # CEBALERT: probably ok for imagen; haven't checked

for package_list in packages_to_state:
    requires = []
    requires+=["%s (%s)"%(package,version) for package,version in package_list.items()]
    setup_args['requires']=requires

#############################################################################################


setup_args.update(dict(
    name='imagen',
    version="2.1.0",
    description='Generic Python library for 0D, 1D, and 2D pattern distributions.',
    long_description=open('README.rst').read() if os.path.isfile('README.rst') else 'Consult README.rst',
    author= "IOAM",
    author_email= "developers@topographica.org",
    maintainer= "IOAM",
    maintainer_email= "developers@topographica.org",
    platforms=['Windows', 'Mac OS X', 'Linux'],
    license='BSD',
    url='http://ioam.github.com/imagen/',
    packages = ["imagen",
                "imagen.transferfn"],
    package_data={},
    classifiers = [
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",        
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries"]
))


def check_pseudo_package(path):
    """
    Verifies that a fake subpackage path for assets (notebooks, svgs,
    pngs etc) both exists and is populated with files.
    """
    if not os.path.isdir(path):
        raise Exception("Please make sure pseudo-package %s exists." % path)
    else:
        assets = os.listdir(path)
        if len(assets) == 0:
            raise Exception("Please make sure pseudo-package %s is populated." % path)


if __name__=="__main__":
    if ('upload' in sys.argv) or ('sdist' in sys.argv):
        import imagen
        imagen.__version__.verify(setup_args['version'])

    setup(**setup_args)
