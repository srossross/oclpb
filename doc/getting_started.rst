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

However: If you are a developer and are check out a version from GitHub. You will require `Cython <http://cython.org>`_. 

If this is the case, when you run `python setup.py ...` you will see an error message like::

    setup.py:66: UserWarning: Cython not installed using pre-cythonized files
      warn("Cython not installed using pre-cythonized files", UserWarning, stacklevel=1)
    Traceback (most recent call last):
      File "setup.py", line 70, in <module>
        raise Exception("Cython is required to build a c extension from a PYX file (solution get cython or checkout a release branch)")
    Exception: Cython is required to build a c extension from a PYX file (solution get cython or checkout a release branch)


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

If you want to test an installed version
you can run run::

    python -c "import opencl; opencl.test()"

.. warning:: Make sure you are not in the source directory.

Otherwise to run tests from the source directory you must run::
    
    python setup.py build_ext --inplace
    python -m unittest discover opencl/

.. note:: you can run these commands before you install as well

