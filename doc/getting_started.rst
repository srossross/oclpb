============================================================
Getting Started
============================================================

easy_install 
--------------------

you can just run::
    
    sudo easy_install oclpb

Prerequisites
--------------------

Only OpenCL!

Download
--------------------

* `Download from stable version from PyPi <http://pypi.python.org/pypi/oclpb>`_
* `Or Download the latest from GitHub <https://github.com/srossross/oclpb/tags>`_

Build
--------

run::

    python setup.py build
    python setup.py install [ --prefix=$SOMPATH ]

Test
--------

run::

    python -m unittest discover opencl/
