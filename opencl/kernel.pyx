
import ctypes
import _ctypes
import sys
from opencl.errors import OpenCLException
from opencl.cl_mem import MemoryObject, DeviceMemoryView

from inspect import isfunction
from opencl.type_formats import refrence, ctype_from_format, type_format, cdefn, cmp_formats
from _cl cimport * 
from opencl.cl_mem import mem_layout

from libc.stdlib cimport malloc, free
from opencl.cl_mem cimport CyMemoryObject_GetID, CyMemoryObject_Check
from cpython cimport PyObject, PyArg_VaParseTupleAndKeywords, Py_INCREF
from opencl.copencl cimport CyDevice_Check, CyDevice_GetID
from opencl.context cimport CyContext_Create

from cpython cimport PyBuffer_FillContiguousStrides
CData = _ctypes._SimpleCData.__base__

def is_string(obj):

    if sys.version_info.major < 3:
        import __builtin__ as builtins
        return isinstance(obj, (str, builtins.unicode))
    else:
        return isinstance(obj, (str,))

def get_code(function):
    if hasattr(function, 'func_code'):
        return function.func_code
    else:
        return function.__code__

DEBUG = False

class contextual_memory(object):
    '''
    Memory 'type' descriptor.
    '''
    qualifier = None
    is_const = False
    def __init__(self, ctype=None, ndim=None, shape=None, flat=False, context=None):
        self.context = context
        self._shape = tuple(shape) if shape else shape
        self.flat = flat
        self.ndim = ndim
        
        if ctype is None:
            self.format = ctype
            self.ctype = ctype
            
        elif is_string(ctype):
            self.format = ctype
            self.ctype = ctype_from_format(ctype)
            
        else:
            self.ctype = ctype
            self.format = type_format(ctype)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.format != other.format:
            return False
        if self.flat != other.flat:
            return False
        
        if not self.flat:
            if not self.ndim == other.ndim:
                return False

        
        return True
    
    def __getattr__(self, attr):
        return getattr(self.ctype, attr)
    
    @property
    def size(self):
        return ctypes.c_size_t

    @property
    def offset(self):
        from opencl.cl_types import cl_uint
        return cl_uint
    @property
    def shape(self):
        from opencl.cl_types import cl_uint4
        return cl_uint4
    
    @property
    def strides(self):
        from opencl.cl_types import cl_uint4
        return cl_uint4
    
    @property
    def nbytes(self):
        nbytes = ctypes.sizeof(self.ctype)
        for item in self._shape:
            nbytes *= item
        return nbytes

    @property
    def array_info(self):
        return mem_layout
    
    def __call__(self, memobj):
        if not isinstance(memobj, MemoryObject):
            raise TypeError("arguemnt must be an instance of MemoryObject")
        cdef cl_mem buffer = CyMemoryObject_GetID(memobj)
        return ctypes.c_voidp(< size_t > buffer)
    
    def ctype_string(self):
        return '%s %s' % (self.qualifier, cdefn(refrence(self.format)))
    
    def derefrence(self):
        '''
        Return the type that this object is a pointer to.
        '''
        return self.ctype
    
    def from_param(self, arg):
        '''
        Return a ctypes.c_void_p from arg. 
        
        :param arg: must be a MemoryObject.
        '''
        if not CyMemoryObject_Check(arg):
            if self.context is None:
                raise TypeError("contextual_memory requires attribute 'context' to be defined for this type of argumetn")
            if not self.is_const:
                if not all(device.host_unified_memory for device in self.context.devices):
                    raise TypeError("A device in the context does not have 'host_unified_memory' can not call kernel with host memory.")
                
            arg = DeviceMemoryView.from_host(self.context, arg, copy=False)
        
        if self.format is not None and hasattr(arg, 'format'):
            if cmp_formats(self.format, arg.format):
                raise TypeError("Expected buffer to be of type %r (got %r)" % (self.format, arg.format))
            
        cdef void * ptr
        
        #FIXME: this should be better #sub-buffer is not supported
#        if arg.context.devices[0].driver_version == '1.0': 
        base = arg.base
        if CyMemoryObject_Check(base):
            arg = base 

        ptr = CyMemoryObject_GetID(arg)
        ctype_ptr = ctypes.c_void_p(< size_t > ptr)
        ctype_ptr._cl_base = arg
        return ctype_ptr
    
    def __repr__(self):
        if self._shape is None:
            return '<memory qualifier=%r format=%r>' % (self.qualifier, self.format)
        else:
            return '<memory qualifier=%r format=%r shape=%s>' % (self.qualifier, self.format, self._shape)

    def __hash__(self):
        return hash((self.qualifier, self.format, self.flat if self.flat else self.ndim))
    
    def _get_array_info(self, obj):
        if isinstance(obj, MemoryObject):
            return obj.array_info
        elif isinstance(obj, local_memory):
            return obj.local_info
        else:
            try:
                view = memoryview(obj)
                ar_inf = self.array_info(0, 0, 0, 0, 0, 0, 0, 0,)
                
                size = 1
                for i in range(view.ndim):
                    size *= view.shape[i]
                    ar_inf[i] = view.shape[i]
                    ar_inf[i + 4] = view.strides[i] / view.itemsize

                ar_inf[3] = size
                
                return ar_inf
            
            except TypeError:
                raise TypeError("Argument must be a cl.MemoryObject or support the new buffer protocol (got %r)" % (obj))
                
            
        
        
        
class global_memory(contextual_memory):
    qualifier = '__global'

class constant_memory(contextual_memory):
    qualifier = '__constant'
    is_const = True
    @property
    def local_strides(self):
        return self.array_info(0, 0, 0, 0, 0, 0, 0, 0,)
    
class local_memory(contextual_memory):
    qualifier = '__local'
    
    @property
    def local_info(self):
        ai = self.array_info(0, 0, 0, 0, 0, 0, 0, 0,)
        
        cdef size_t ndim = len(self._shape)
        
        cdef Py_ssize_t shape[4]
        cdef Py_ssize_t strides[4]
        
        ai[3] = 1
        for i, item in enumerate(self._shape):
            shape[i] = item
            ai[i] = item
            ai[3] = ai[3] * item
        
        PyBuffer_FillContiguousStrides(ndim, shape, strides, 1, 'C')        
        
        
        for i in range(ndim):
            ai[4 + i] = strides[i]
            
        return ai
    
set_kerne_arg_errors = {
    CL_INVALID_KERNEL : 'kernel is not a valid kernel object.',
    CL_INVALID_ARG_INDEX :'arg_index is not a valid argument index.',
    CL_INVALID_ARG_VALUE : 'arg_value specified is not a valid value.',
    CL_INVALID_MEM_OBJECT : 'The specified arg_value is not a valid memory object.',
    CL_INVALID_SAMPLER : 'The specified arg_value is not a valid sampler object.',
    CL_INVALID_ARG_SIZE :('arg_size does not match the size of the data type for an ' 
                          'argument that is not a memory object or if the argument is a memory object and arg_size')
}

class _Undefined: pass

def call_with_used_args(func, argnames, arglist):
    '''
    Call a function with argument.but only the arguments with names found in 
    `func.func_code.co_varnames`
    
    :param func: function to call
    :param argnames: names of the arguments in `arglist` 
    :param arglist: arguements to call with
     
    '''
    
    code = get_code(func)
    func_args = code.co_varnames[:code.co_argcount]
    
    if argnames is None:
        args = arglist
    else:
        args = [arg for name, arg in zip(argnames, arglist) if name in func_args]
        
    result = func(*args)
    return result
    
def parse_args(name, args, kwargs, argnames, defaults):
    '''
    parse_args(name, args, kwargs, argnames, defaults) -> list
    
    similar to python's c api parse args. 
    '''
    narg_names = len(argnames)
    nargs = len(args)
    
    if nargs > narg_names:
        raise TypeError("%s() takes at most %i argument(s) (%i given)" % (name, narg_names, nargs))
    
    if not defaults: defaults = ()
    default_idx = narg_names - len(defaults)
    
    result = [_Undefined]*narg_names
    
    arg_set = set(argnames[:nargs])
    kw_set = set(kwargs)
    overlap = kw_set.intersection(arg_set)
    if overlap:
        raise TypeError("%s() got multiple values for keyword argument(s) %r" % (name, overlap))
    
    extra = kw_set - set(argnames)
    if extra:
        raise TypeError("%s() got unexpected keyword argument(s) %r" % (name, extra))
    
    result[default_idx:] = defaults
    result[:nargs] = args
    
    expected_kw = argnames[nargs:default_idx]
    cdef int i
    for i in range(nargs, default_idx):
        required_keyword = argnames[i] 
        if required_keyword not in kwargs:
            raise TypeError("%s() takes at least %i argument(s) (%i given)" % (name, default_idx, nargs))
        
        result[i] = kwargs[required_keyword]

    for i in range(default_idx, narg_names):
        keyword = argnames[i]
        result[i] = kwargs.get(keyword, result[i])
        
    return tuple(result)

cdef class Kernel:
    '''
    openCl kernel object.
    
    A kernel object encapsulates a specific __kernel function declared in a 
    program and the argument values to be used when executing this __kernel function.
    
    
    '''
    cdef cl_kernel kernel_id
    cdef object _argtypes 
    cdef object _argnames 
    cdef public object __defaults__
    cdef public object global_work_size
    cdef public object global_work_offset
    cdef public object local_work_size
    
    def __cinit__(self):
        self.kernel_id = NULL
        self._argtypes = None
        self._argnames = None
        self.global_work_size = None
        self.global_work_offset = None
        self.local_work_size = None

    def __dealloc__(self):
        
        if self.kernel_id != NULL:
            clReleaseKernel(self.kernel_id)
            
        self.kernel_id = NULL
        
    def __init__(self):
        raise TypeError("kernel can not be constructed from python")
    
    property argtypes:
        '''
        Assign a tuple of ctypes types to specify the argument types that the function accepts 
        
        len(argtypes) must equal kernel.nargs.
        
        It is now possible to put items in argtypes which are not ctypes types, but each item 
        must have a from_param() method which returns a value usable as 
        argument (integer, string, ctypes instance). This allows to define 
        adapters that can adapt custom objects as function parameters.
        
        .. seealso:: http://docs.python.org/library/ctypes.html#type-conversions  
        '''

        def __get__(self):
            return self._argtypes
        
        def __set__(self, value):
            self._argtypes = tuple(value)
            if len(self._argtypes) != self.nargs:
                raise TypeError("argtypes must have %i values (got %i)" % (self.nargs, len(self._argtypes)))

    property argnames:
        '''
        Get or set the argument names. 
        len(argnames) must equal kernel.nargs  
        '''
        
        def __get__(self):
            return self._argnames
        
        def __set__(self, value):
            self._argnames = tuple(value)
            if len(self._argnames) != self.nargs:
                raise TypeError("argnames must have %i values (got %i)" % (self.nargs, len(self._argnames)))
            
    property nargs:
        'Number of arguments that this kernel takes'
        def __get__(self):
            cdef cl_int err_code
            cdef cl_uint nargs

            err_code = clGetKernelInfo(self.kernel_id, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), & nargs, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            return nargs

#    property program:
#        'the program that this kernel was created in'
#        def __get__(self):
#            cdef cl_int err_code
#            cdef cl_program program_id
#
#            err_code = clGetKernelInfo(self.kernel_id, CL_KERNEL_PROGRAM, sizeof(cl_program), & program_id, NULL)
#            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
#            
#            
#            return CyProgram_Create(program_id)

    property context:
        'The context this kernel was created with.'
        def __get__(self):
            cdef cl_int err_code
            cdef cl_context context_id

            err_code = clGetKernelInfo(self.kernel_id, CL_KERNEL_CONTEXT, sizeof(cl_context), & context_id, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            return CyContext_Create(context_id)

    def work_group_size(self, device):
        '''
        kernel.work_group_size(device)
        
        This provides a mechanism for the application to query the maximum 
        work-group size that can be used to execute a kernel on a specific device 
        given by device.  The OpenCL implementation uses the resource 
        requirements of the kernel (register usage etc.) to determine what this workgroup size should be. 
        '''
        if not CyDevice_Check(device):
            raise TypeError("expected argument to be a device")
        
        cdef cl_device_id device_id = CyDevice_GetID(device) 
        cdef cl_int err_code
        cdef cl_context context_id
        cdef size_t value

        err_code = clGetKernelWorkGroupInfo(self.kernel_id, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), & value, NULL)
        if err_code != CL_SUCCESS: raise OpenCLException(err_code)
        
        return value

    def preferred_work_group_size_multiple(self, device):
        '''
        kernel.preferred_work_group_size_multiple(device)
        
        Returns the preferred multiple of workgroup size for launch.  This is a 
        performance hint. Specifying a workgroup size that is not a multiple of the 
        value returned by this query as the value of the local work size argument to 
        clEnqueueNDRangeKernel will not fail to enqueue the kernel for execution 
        unless the work-group size specified is larger than the device maximum.
        '''
        if not CyDevice_Check(device):
            raise TypeError("expected argument to be a device")
        
        cdef cl_device_id device_id = CyDevice_GetID(device) 
        cdef cl_int err_code
        cdef cl_context context_id
        cdef size_t value

        err_code = clGetKernelWorkGroupInfo(self.kernel_id, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), & value, NULL)
        if err_code != CL_SUCCESS: raise OpenCLException(err_code)
        
        return value

    def private_mem_size(self, device):
        '''
        kernel.private_mem_size(device)
        
        Returns the minimum amount of private  memory, in bytes, used by each workitem in the kernel.  This value may 
        include any private memory needed by  an implementation to execute the kernel,
        including that used by the language  built-ins and variable declared inside the 
        kernel with the __private qualifier.
        '''
        if not CyDevice_Check(device):
            raise TypeError("expected argument to be a device")
        
        cdef cl_device_id device_id = CyDevice_GetID(device) 
        cdef cl_int err_code
        cdef cl_context context_id
        cdef size_t value

        err_code = clGetKernelWorkGroupInfo(self.kernel_id, device_id, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(size_t), & value, NULL)
        if err_code != CL_SUCCESS: raise OpenCLException(err_code)
        
        return value

    def local_mem_size(self, device):
        '''
        kernel.local_mem_size(device)
        
        Returns the amount of local memory in bytes being used by a kernel. This
        includes local memory that may be needed by an implementation to execute 
        the kernel, variables declared inside the  kernel with the __local address 
        qualifier and local memory to be  allocated for arguments to the kernel 
        declared as pointers with the __local address qualifier and whose size is 
        specified with clSetKernelArg
        '''
        
        if not CyDevice_Check(device):
            raise TypeError("expected argument to be a device")
        
        cdef cl_device_id device_id = CyDevice_GetID(device) 
        cdef cl_int err_code
        cdef cl_context context_id
        cdef size_t value

        err_code = clGetKernelWorkGroupInfo(self.kernel_id, device_id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t), & value, NULL)
        if err_code != CL_SUCCESS: raise OpenCLException(err_code)
        
        return value

    def compile_work_group_size(self, device):
        '''
        kernel.compile_work_group_size(device)
        
        Returns the work-group size specified by the __attribute__((reqd_work_group_size(X, Y, Z))) qualifier.
        '''
        if not CyDevice_Check(device):
            raise TypeError("expected argument to be a device")
        
        cdef cl_device_id device_id = CyDevice_GetID(device) 
        cdef cl_int err_code
        cdef cl_context context_id
        cdef size_t value[3]

        err_code = clGetKernelWorkGroupInfo(self.kernel_id, device_id, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(size_t) * 3, < void *> value, NULL)
        if err_code != CL_SUCCESS: raise OpenCLException(err_code)
        
        return (value[0], value[1], value[2])


    property name:
        'The name of this kernel'
        
        def __get__(self):
            cdef cl_int err_code
            cdef size_t nbytes
            cdef char * name = NULL
            
            err_code = clGetKernelInfo(self.kernel_id, CL_KERNEL_FUNCTION_NAME, 0, NULL, & nbytes)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            name = < char *> malloc(nbytes + 1)
            
            err_code = clGetKernelInfo(self.kernel_id, CL_KERNEL_FUNCTION_NAME, nbytes, name, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            name[nbytes] = 0
            pyname = name.decode('UTF-8')
            free(name)
            
            return pyname
        
    def __repr__(self):
        return '<Kernel %s nargs=%r>' % (self.name, self.nargs)
    
    def set_args(self, *args, **kwargs):
        '''
        kernel.set_args(self, *args, **kwargs)
        Set the arguments for this kernel
        '''
        global DEBUG
        if self._argtypes is None:
            raise TypeError("argtypes must be set before calling ")
        
        argnames = range(self.nargs) if self._argnames is None else self._argnames
        defaults = [] if  self.__defaults__ is None else self.__defaults__
         
        arglist = parse_args(self.name, args, kwargs, argnames, defaults)
        
        cdef cl_int err_code
        cdef size_t arg_size
        cdef size_t tmp
        cdef void * arg_value
        cdef cl_mem mem_id
        
        cargs = {}
        for arg_index, (argtype, arg) in enumerate(zip(self._argtypes, arglist)):
            
            if isinstance(arg, local_memory):
                arg_size = arg.nbytes
                arg_value = NULL
            else:
                try:
                    carg = argtype.from_param(arg)
                    cargs[argnames[arg_index]] = carg
                except TypeError as err:
                    if DEBUG: raise
                    if self._argnames is None:
                        raise TypeError("argument at pos %i expected type to be %r (got %r) msg='%s'" % (arg_index, argtype, arg, err))
                    else:
                        raise TypeError("argument %r (pos %i) expected type to be %r (got %r) msg='%s'" % (argnames[arg_index], arg_index, argtype, arg, err))
                
                if not isinstance(carg, CData):
                    carg = argtype(arg)
                    
                arg_size = ctypes.sizeof(carg)
                tmp = < size_t > ctypes.addressof(carg)
                arg_value = < void *> tmp
                
            err_code = clSetKernelArg(self.kernel_id, arg_index, arg_size, arg_value)
            if err_code != CL_SUCCESS:
                if err_code == CL_INVALID_ARG_SIZE:
                    msg = 'argument %r had size of %i (please check kernel source)' %(argnames[arg_index], arg_size)
                    raise OpenCLException(err_code, msg=msg)

                raise OpenCLException(err_code, set_kerne_arg_errors)
            
        #Keeping cargs alive until enqueue_nd_range_kernel in case cargs is a from_host memory object.
        #Otherwise it will be garbage collected. 
        return arglist, cargs
         
    def __call__(self, queue, *args, global_work_size=None, global_work_offset=None, local_work_size=None, wait_on=(), **kwargs):
        '''
        kernel(queue, *args, global_work_size=None, global_work_offset=None, local_work_size=None, wait_on=(), **kwargs)
        
        Set a kernels args and enqueue an nd_range_kernel to the queue.
        '''
        arglist, cargs = self.set_args(*args, **kwargs)
        
        
        if global_work_size is None:
            if isfunction(self.global_work_size):
                global_work_size = call_with_used_args(self.global_work_size, self.argnames, arglist)
            elif self.global_work_size is None:
                raise TypeError("missing required keyword arguement 'global_work_size'")
            else:
                global_work_size = self.global_work_size

        if global_work_offset is None:
            if isfunction(self.global_work_offset):
                global_work_offset = call_with_used_args(self.global_work_offset, self.argnames, arglist)
            else:
                global_work_offset = self.global_work_offset

        if local_work_size is None:
            if isfunction(self.local_work_size):
                local_work_size = call_with_used_args(self.local_work_size, self.argnames, arglist)
            else:
                local_work_size = self.local_work_size

        return queue.enqueue_nd_range_kernel(self, len(global_work_size), global_work_size, global_work_offset, local_work_size, wait_on)
    
#===============================================================================
# API
#===============================================================================

cdef api cl_kernel KernelFromPyKernel(object py_kernel):
    cdef Kernel kernel = < Kernel > py_kernel
    return kernel.kernel_id

cdef api object KernelAsPyKernel(cl_kernel kernel_id):
    cdef Kernel kernel = < Kernel > Kernel.__new__(Kernel)
    kernel.kernel_id = kernel_id
    clRetainKernel(kernel_id)
    return kernel
