=========================
OpenCL for Python API
=========================


.. automodule:: opencl

opencl.Platform
----------------------

.. autoclass:: Platform
    :members:
    :undoc-members:

    
opencl.Device
--------------------------
    
.. autoclass:: Device
    :members:
    :undoc-members: 
    
    .. data:: DEFAULT
    
        flag: device type default.
    
    .. data:: ALL
    
        flag: for all devices
    
    .. data:: CPU
    
        flag: for all CPU devices
        
    .. data:: GPU
    
        flag: for all GPU devices
        
opencl.Event
--------------------------
    
.. autoclass:: Event
    :members:
    :undoc-members: 
   
opencl.UserEvent
--------------------------
    
.. autoclass:: UserEvent()
    :members:
    :undoc-members: 
   

opencl.Program
--------------------------
    
.. autoclass:: Program(context, source=None, binaries=None, devices=None)
    :members:
    :undoc-members: 
   

opencl.MemoryObject
--------------------------

.. autoclass:: MemoryObject
    :members:
    :undoc-members: 
   
opencl.DeviceMemoryView
--------------------------

.. autoclass:: DeviceMemoryView
    :members:
    :undoc-members: 
   
opencl.ImageFormat
--------------------------

.. autoclass:: ImageFormat
    :members:
    :undoc-members: 
   
opencl.Image
--------------------------

.. autoclass:: Image
    :members:
    :undoc-members: 
   
opencl.ContextProperties
--------------------------

.. autoclass:: ContextProperties
    :members:
    :undoc-members: 
   
opencl.Context
--------------------------

.. autoclass:: Context(devices=(), device_type=cl.Device.DEFAULT, properties=None, callback=None)
    :members:
    :undoc-members: 
   
opencl.OpenCLException
--------------------------

.. autoclass:: OpenCLException
    :members:
    :undoc-members: 
   
opencl.contextual_memory
--------------------------

.. autoclass:: contextual_memory(ctype=None, shape=None, flat=False)
    :members:
    :undoc-members: 
   
opencl.global_memory
--------------------------

.. autoclass:: global_memory(ctype=None, shape=None, flat=False)
    :members:
    :undoc-members: 

opencl.Kernel
--------------------------

.. autoclass:: Kernel()
    :members:
    :undoc-members: 
   
  
opencl.Queue
--------------------------

.. autoclass:: Queue()
    :members:
    :undoc-members: 

  
Functions
--------------------------

.. autofunction empty(context, shape, ctype='B')

.. autofunction empty_image(context, shape, image_format)

.. autofunction broadcast(DeviceMemoryView view, shape)


.. automodule opencl.type_formats
