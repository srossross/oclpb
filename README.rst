Welcome to OpenCL for Python's documentation!
===============================================

This is yet another set of Python bindings for OpenCL.


.. warning:: This project currently is in a beta release state. 


Features:
+++++++++++

* Python 2 and Python 3 compatibility.
* Supports OpenCL 1.1 
* Discoverable properties and methods:
    No more **ctx.get_info(pyopencl.context_info.DEVICES)** just do **ctx.devices** 
* Tight integration with `ctypes <http://docs.python.org/library/ctypes.html>`_::
    
    import opencl as cl
    from ctypes import c_float
    ctx =  cl.Context()
    a = cl.empty(ctx, [2, 3], ctype=c_float)
     
* Call kernels like a python function with defaults and keyword arguments::
    
    import opencl as cl
    from ctypes import c_float, c_int
    
    source = '__kernel void foo(__global float*a, int b, float c) ...'
    ...
    # Create a program and context
    
    foo = program.foo
    foo.argnames = 'a', 'b', 'c'
    foo.argtypes = cl.global_memory(c_float, ndim=2), c_int, c_float
    # Equivalent to def foo(a, b=1, c=2.0):
    foo.__defaults__ = 1, 2.0
    
    event = foo(queue, a)
    
* Memory objects support indexing and slicing::
    
    mem2 = memobj[:, 1, :-1]
    
Links:
+++++++++++

* `Homepage <http://srossross.github.com/oclpb/>`_
* `Issue Tracker <https://github.com/srossross/oclpb/issues/>`_


* `Development documentation <http://srossross.github.com/oclpb/develop/>`_
* `PyPi <http://pypi.python.org/pypi/opencl-for-python/>`_
* `Github <https://github.com/srossross/oclpb/>`_
* `OpenCL 1.1 spec <http://www.khronos.org/registry/cl/specs/opencl-1.0.29.pdf>`_

* Also please check out `CLyther <http://srossross.github.com/Clyther>`_
