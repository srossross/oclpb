'''

'''
import weakref
import struct
import ctypes
from opencl.type_formats import refrence, ctype_from_format, type_format, cdefn
from opencl.errors import OpenCLException, BuildError

from libc.stdlib cimport malloc, free 
from libc.stdio cimport printf
from _cl cimport * 
from cpython cimport PyObject, Py_DECREF, Py_INCREF, PyBuffer_IsContiguous, PyBuffer_FillContiguousStrides
from cpython cimport Py_buffer, PyBUF_SIMPLE, PyBUF_STRIDES, PyBUF_ND, PyBUF_FORMAT, PyBUF_INDIRECT, PyBUF_WRITABLE

cdef extern from "Python.h":
    void PyEval_InitThreads()

MAGIC_NUMBER = 0xabc123

PyEval_InitThreads()

cpdef get_platforms():
    '''
    Return a list of the platforms connected to the host.
    '''
    cdef cl_uint num_platforms
    cdef cl_platform_id plid
    
    ret = clGetPlatformIDs(0, NULL, & num_platforms)
    if ret != CL_SUCCESS:
        raise OpenCLException(ret)
    cdef cl_platform_id * cl_platform_ids = < cl_platform_id *> malloc(num_platforms * sizeof(cl_platform_id *))
    
    ret = clGetPlatformIDs(num_platforms, cl_platform_ids, NULL)
    
    if ret != CL_SUCCESS:
        free(cl_platform_ids)
        raise OpenCLException(ret)
    
    platforms = []
    for i in range(num_platforms):
        plat = <Platform> Platform.__new__(Platform)
        plat.platform_id = cl_platform_ids[i]
        platforms.append(plat)
        
    free(cl_platform_ids)
    return platforms
    

cdef class Platform:
    '''
    opencl.Platform not constructible.
    
    Use  opencl.get_platforms() to get a list of connected platoforms.
    '''
    cdef cl_platform_id platform_id
    
    def __cinit__(self):
        pass
    
    def __init__(self):
        raise Exception("Can not create a platform: use opencl.get_platforms()")
    
    def __repr__(self):
        return '<opencl.Platform name=%r profile=%r>' % (self.name, self.profile,)

    
    cdef get_info(self, cl_platform_info info_type):
        cdef size_t size
        cdef cl_int err_code
        err_code = clGetPlatformInfo(self.platform_id,
                                   info_type, 0,
                                   NULL, & size)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        cdef char * result = < char *> malloc(size * sizeof(char *))
        
        err_code = clGetPlatformInfo(self.platform_id,
                                   info_type, size,
                                   result, NULL)
        
        if err_code != CL_SUCCESS:
            free(result)
            raise OpenCLException(err_code)
        
        cdef bytes a_python_byte_string = result
        free(result)
        return a_python_byte_string

    property profile:
        '''
        return the plafrom profile info
        '''
        def __get__(self):
            return self.get_info(CL_PLATFORM_PROFILE)

    property version:
        '''
        return the version string of the platform
        '''
        def __get__(self):
            return self.get_info(CL_PLATFORM_VERSION)
        
    property name:
        'platform name'
        def __get__(self):
            return self.get_info(CL_PLATFORM_NAME)

    property vendor:
        'platform vendor'
        def __get__(self):
            return self.get_info(CL_PLATFORM_VENDOR)

    property extensions:
        'platform extensions as a string'
        def __get__(self):
            return self.get_info(CL_PLATFORM_EXTENSIONS)

    property devices:
        'list of all devices attached to this platform'
        def __get__(self):
            return self.get_devices()

    def  get_devices(self, cl_device_type device_type=CL_DEVICE_TYPE_ALL):
        '''
        plat.get_devices(device_type=opencl.Device.ALL)
        
        return a list of devices by type.
        '''
        cdef cl_int err_code
           
        cdef cl_uint num_devices
        err_code = clGetDeviceIDs(self.platform_id, device_type, 0, NULL, & num_devices)
            
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        cdef cl_device_id * result = < cl_device_id *> malloc(num_devices * sizeof(cl_device_id *))
        
        err_code = clGetDeviceIDs(self.platform_id, device_type, num_devices, result, NULL)
        
        devices = []
        for i in range(num_devices):
            device = <Device> Device.__new__(Device)
            device.device_id = result[i]
            devices.append(device)
            
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        return devices
        
    
    def __hash__(self):
        return < size_t > self.platform_id

    def __richcmp__(Platform self, other, op):
        
        if not isinstance(other, Platform):
            return NotImplemented
        
        if op == 2:
            return self.platform_id == CyPlatform_GetID(other)
        else:
            return NotImplemented

cdef class Device:
    '''
    A device is a collection of compute units.  A command-queue is used to queue 
    commands to a device.  Examples of commands include executing kernels, or reading and writing 
    memory objects. 
    
    OpenCL devices typically correspond to a GPU, a multi-core CPU, and other 
    processors such as DSPs and the Cell/B.E. processor.
    
    '''
    ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR
    ALL = CL_DEVICE_TYPE_ALL
    CPU = CL_DEVICE_TYPE_CPU
    DEFAULT = CL_DEVICE_TYPE_DEFAULT
    GPU = CL_DEVICE_TYPE_GPU
    
    DEV_TYPE_MAP = {
        'ACCELERATOR' : CL_DEVICE_TYPE_ACCELERATOR,
        'ALL' : CL_DEVICE_TYPE_ALL,
        'CPU' : CL_DEVICE_TYPE_CPU,
        'DEFAULT' : CL_DEVICE_TYPE_DEFAULT,
        'GPU' : CL_DEVICE_TYPE_GPU,
                   }
    
    cdef cl_device_id device_id

    def __cinit__(self):
        pass
    
    def __init__(self):
        raise Exception("opencl.Device object can not be constructed.")
    
    def __repr__(self):
        return '<opencl.Device name=%r type=%r>' % (self.name, self.type,)
    
    def __hash__(Device self):
        
        cdef size_t hash_id = < size_t > self.device_id

        return int(hash_id)
    
    def __richcmp__(Device self, other, op):
        
        if not isinstance(other, Device):
            return NotImplemented
        
        if op == 2:
            return self.device_id == (< Device > other).device_id
        else:
            return NotImplemented
            
    property platform:
        '''
        return the platform this device is associated with.
        '''
        def __get__(self):
            cdef cl_int err_code
            cdef cl_platform_id plat_id
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), < void *>& plat_id, NULL)
                
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return CyPlatform_Create(plat_id)
        
    property type:
        'return device type: one of [Device.DEFAULT, Device.ALL, Device.GPU or Device.CPU]'
        def __get__(self):
            cdef cl_int err_code
            cdef cl_device_type dtype
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), < void *>& dtype, NULL)
                
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return dtype

    property has_image_support:
        'test if this device supports the openc.Image class'
        def __get__(self):
            cdef cl_int err_code
            cdef cl_bool result
            
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), < void *>& result, NULL)
                
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return True if result else False

    property name:
        'the name of this device'
        def __get__(self):
            cdef size_t size
            cdef cl_int err_code
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_NAME, 0, NULL, & size)
            
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            cdef char * result = < char *> malloc(size * sizeof(char *))
            
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_NAME, size * sizeof(char *), < void *> result, NULL)

            if err_code != CL_SUCCESS:
                free(result)
                raise OpenCLException(err_code)
            
            cdef bytes a_python_byte_string = result
            free(result)
            return a_python_byte_string

    property queue_properties:
        '''
        return queue properties as a bitfield
        
        see also `has_queue_out_of_order_exec_mode` and `has_queue_profiling`
        '''
        def __get__(self):
            cdef size_t size
            cdef cl_int err_code
            cdef cl_command_queue_properties result
            
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), & result, NULL)
            
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return result 
        
    property has_queue_out_of_order_exec_mode:
        'test if this device supports out_of_order_exec_mode for queues'
        def __get__(self):
            return bool((<cl_command_queue_properties> self.queue_properties) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)

    property has_queue_profiling:
        'test if this device supports profiling for queues'
        def __get__(self):
            return bool((<cl_command_queue_properties> self.queue_properties) & CL_QUEUE_PROFILING_ENABLE)
        
    property has_native_kernel:
        'test if this device supports native python kernels'
        def __get__(self):
            cdef cl_int err_code
            cdef cl_device_exec_capabilities result
            
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(cl_device_exec_capabilities), & result, NULL)
            
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return True if result & CL_EXEC_NATIVE_KERNEL else False 

    property vendor_id:
        'return the vendor ID'
        def __get__(self):
            cdef cl_int err_code
            cdef cl_uint value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_VENDOR_ID, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value
        
    property max_compute_units:
        '''
        The number of parallel compute cores on the OpenCL device.  
        The minimum value is 1.
        '''
        def __get__(self):
            cdef cl_int err_code
            cdef cl_uint value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property max_work_item_dimensions:
        '''
        Maximum dimensions that specify the  global and local work-item IDs used 
        by the data parallel execution model. (Refer to clEnqueueNDRangeKernel).
          
        The minimum value is 3.
        '''
        def __get__(self):
            cdef cl_int err_code
            cdef cl_uint value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property max_clock_frequency:
        '''
        return the clock frequency. 
        '''
        def __get__(self):
            cdef cl_int err_code
            cdef cl_uint value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property address_bits:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_uint value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_ADDRESS_BITS, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value
        
    property max_read_image_args:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_uint value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property max_write_image_args:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_uint value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property global_mem_size:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_ulong value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value
        
    property max_mem_alloc_size:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_ulong value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property max_const_buffer_size:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_ulong value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property has_local_mem:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_device_local_mem_type value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value == CL_LOCAL

    property local_mem_size:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_ulong value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property host_unified_memory:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_bool value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return bool(value)

    property available:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_bool value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_AVAILABLE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return bool(value)

    property compiler_available:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_bool value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_COMPILER_AVAILABLE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return bool(value)

    property max_work_item_sizes:
        '''
        Maximum number of work-items that  can be specified in each dimension to  `opencl.Queue.enqueue_nd_range_kernel`.
          
        :returns: n entries, where n is the value returned by the query for  `opencl.Device.max_work_item_dimensions`
        '''
        def __get__(self):
            cdef cl_int err_code
            cdef size_t dims = self.max_work_item_dimensions
            cdef size_t nbytes = sizeof(size_t) * dims
            cdef size_t * value = < size_t *> malloc(nbytes)
            
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, nbytes, < void *> value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            result = [value[i] for i in range(dims)]
            free(value)
            
            return result

    property max_work_group_size:
        def __get__(self):
            cdef cl_int err_code
            cdef size_t value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property profiling_timer_resolution:
        def __get__(self):
            cdef cl_int err_code
            cdef size_t value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property max_parameter_size:
        def __get__(self):
            cdef cl_int err_code
            cdef size_t value = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(value), < void *>& value, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return value

    property max_image2d_shape:
        def __get__(self):
            cdef cl_int err_code
            cdef size_t w = 0
            cdef size_t h = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(w), < void *>& w, NULL)
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(h), < void *>& h, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return [w, h]

    property max_image3d_shape:
        def __get__(self):
            cdef cl_int err_code
            cdef size_t w = 0
            cdef size_t h = 0
            cdef size_t d = 0
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(w), < void *>& w, NULL)
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(h), < void *>& h, NULL)
            err_code = clGetDeviceInfo(self.device_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(d), < void *>& d, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            return [w, h, d]

    cdef get_info(self, cl_device_info info_type):
        cdef size_t size
        cdef cl_int err_code
        err_code = clGetDeviceInfo(self.device_id, info_type, 0, NULL, & size)
        if err_code != CL_SUCCESS: raise OpenCLException(err_code)
        
        cdef char * result = < char *> malloc(size * sizeof(char *))
        
        err_code = clGetDeviceInfo(self.device_id, info_type, size, result, NULL)
        
        if err_code != CL_SUCCESS:
            free(result)
            raise OpenCLException(err_code)
        
        cdef bytes a_python_byte_string = result
        free(result)
        return a_python_byte_string

    property driver_version:
        def __get__(self):
            return self.get_info(CL_DRIVER_VERSION)

    property version:
        def __get__(self):
            return self.get_info(CL_DEVICE_PROFILE)
        
    property profile:
        def __get__(self):
            return self.get_info(CL_DEVICE_VERSION)
        
    property extensions:
        def __get__(self):
            return self.get_info(CL_DEVICE_EXTENSIONS).split()

        
## API FUNCTIONS #### #### #### #### #### #### #### #### #### #### ####
## ############# #### #### #### #### #### #### #### #### #### #### ####
#===============================================================================
# 
#===============================================================================

cdef api cl_platform_id CyPlatform_GetID(object py_platform):
    cdef Platform platform = < Platform > py_platform
    return platform.platform_id

cdef api object CyPlatform_Create(cl_platform_id platform_id):
    cdef Platform platform = < Platform > Platform.__new__(Platform)
    platform.platform_id = platform_id
    return platform

#===============================================================================
# 
#===============================================================================

cdef api int CyDevice_Check(object py_device):
    return isinstance(py_device, Device)

cdef api cl_device_id CyDevice_GetID(object py_device):
    cdef Device device = < Device > py_device
    return device.device_id

cdef api object CyDevice_Create(cl_device_id device_id):
    cdef Device device = < Device > Device.__new__(Device)
    device.device_id = device_id
    return device



