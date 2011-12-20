==============================
Differences from PyOpenCL
==============================

import opencl
--------------

PyOpenCL::

    import pyopencl as cl
    
OpenCL for Python ::

    import opencl as cl
    
    
Properties and Flags
----------------------

In PyOpenCL one would would write::

    ctx = pyopencl.Context()
    devices = ctx.get_info(pyopencl.context_info.DEVICES)
    
OpenCL for Python::

    ctx = pyopencl.Context()
    devices = ctx.devices


MemoryObjects
----------------------

OpenCL for Python does not require numpy and has no external dependencies. It only relies hevily on the memoryview object and ctypes data structures.
 
The main memory object in OpenCL for Python is a :class:`opencl.DeviceMemoryView`. a device memoryview supports slicing and copying. 
To create a view of device memory you can use :func:`opencl.from_host`  or :func:`opencl.empty`

example::

    na = np.arange(20)
    a =  opencl.from_host(na, copy=True)
    
    # and  
    b =  opencl.empty(shape, ctype='f')
    
The argument `ctype` may be a valid ctype or subclass from the ctypes module or a valid data format descriptor.


.. seealso::

    ctypes module documentation.
        http://docs.python.org/library/ctypes.html
        
    Examples of data format descriptions.
        http://www.python.org/dev/peps/pep-3118/#examples-of-data-format-descriptions
    



Kernels
----------------------

Kernels may follow the ctypes convention. and define an `argtypes` attribute. `argnames` and `__defaults__` may also be defined. 
 
example::

    program = cl.Program( '''__kernel void foo(__global *a, float b, int x) ... ''').build()
    
    foo = program.foo
    foo.argnames = 'a', 'b', 'x'
    foo.argtypes = cl.global_memory('f'), cl.cl_float, cl.cl_int
    
    #global_work_size is either a function or sequence of integers.
    foo.global_work_size = lambda a: a.shape
    
    #Equivalent to b=2.0, x=1 
    foo.__defaults__ = 10.0, 1
    
    #The following invocations of foo are all equivalent.
    foo(queue, a=cl_memory)
    foo(queue, a=cl_memory, x=1, b=10)
    foo(queue, cl_memory, 10, 1)
    foo(queue, a=cl_memory, b=10, x=1, global_work_size=[cl_memory.size])
    

CommandQueue
-----------------------

Can be referred to as :class:`opencl.Queue`.

all `enqueue_*` functions are now methods on the :class:`opencl.Queue` class 


