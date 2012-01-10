
from .copencl import Platform, get_platforms, Device
from .program import Program
from .event import UserEvent, Event

from .context import Context, ContextProperties
from .kernel import contextual_memory, global_memory, local_memory, constant_memory, Kernel
from .queue import Queue
from .cl_mem import MemoryObject
from .cl_mem import DeviceMemoryView, empty
from .cl_mem import mem_layout, broadcast
from .cl_mem import empty_image, Image, ImageFormat
from .errors import OpenCLException
import sys

try:
    import opencl.clgl as gl
except:
    gl = None


from_host = DeviceMemoryView.from_host


from .cl_types import *

CommandQueue = Queue


def get_include():
    '''
    Return the directory that contains the OpenCL for Python \*.h header files.
    
    Extension modules that need to compile against OpenCL for Python should use this
    function to locate the appropriate include directory.
    
    Notes
    -----
    When using ``distutils``, for example in ``setup.py``.
    ::
    
        import opencl as cl
        ...
        Extension('extension_name', ...
                include_dirs=[cl.get_include()])

    '''
    from os.path import dirname
    return dirname(__file__)

def test(stream=sys.stdout, descriptions=True, verbosity=2, failfast=False, buffer=False):
    import unittest as _unit
    import os as _os
    star_dir = _os.path.dirname(__file__)
    test_suite = _unit.defaultTestLoader.discover(star_dir)
    runner = _unit.TextTestRunner(stream, descriptions, verbosity, failfast, buffer)
    runner.run(test_suite)
    