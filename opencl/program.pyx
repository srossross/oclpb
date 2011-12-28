import weakref
import struct
import ctypes
from opencl.errors import OpenCLException, BuildError

from libc.stdlib cimport malloc, free 
from libc.stdio cimport printf
from _cl cimport * 

from opencl.copencl cimport CyDevice_Create, CyDevice_Check, CyDevice_GetID
from opencl.context cimport CyContext_GetID, CyContext_Create, CyContext_Check
from opencl.kernel cimport KernelAsPyKernel

clCreateKernel_errors = {}

cdef class Program:
    '''
    
    Create an opencl program.
    
    :param context: opencl.Context object.
    :param source: program source to compile.
    :param binaries: dict of pre-compiled binaries. of the form {device:bytes, ..}
    :param devices: list of devices to compile on.
    
    To get a kernel do `program.name` or `program.kernel('name')`.
    
    '''
    
    NONE = CL_BUILD_NONE
    ERROR = CL_BUILD_ERROR
    SUCCESS = CL_BUILD_SUCCESS
    IN_PROGRESS = CL_BUILD_IN_PROGRESS
    
    cdef cl_program program_id
    
    def __cinit__(self):
        self.program_id = NULL
    
    def __dealloc__(self):
        if self.program_id != NULL:
            clReleaseProgram(self.program_id)
        self.program_id = NULL
        
    def __init__(self, context, source=None, binaries=None, devices=None):
        
        cdef char * strings
        cdef cl_int err_code
        
        if not CyContext_Check(context):
            raise TypeError("argument 'context' must be a valid opencl.Context object")
        cdef cl_context ctx = CyContext_GetID(context)
        
        cdef cl_uint num_devices
        cdef cl_device_id * device_list
        cdef size_t * lengths 
        cdef unsigned char ** bins
        cdef cl_int * binary_status
        
        
        if source is not None:
            byte_source = source.encode()
            strings = byte_source
            
            self.program_id = clCreateProgramWithSource(ctx, 1, & strings, NULL, & err_code)
            
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)

        elif binaries is not None:
            
            num_devices = len(binaries)
            
            device_list = < cl_device_id *> malloc(sizeof(cl_device_id) * num_devices)
            lengths = < size_t *> malloc(sizeof(size_t) * num_devices)
            bins = < unsigned char **> malloc(sizeof(unsigned char *) * num_devices)
            binary_status = < cl_int *> malloc(sizeof(cl_int) * num_devices)
            
            try:
                for i, (device, binary) in enumerate(binaries.items()):
                    
                    if not CyDevice_Check(device):
                        raise TypeError("argument binaries must be a dict of device:binary pairs")
                    
                    device_list[i] = CyDevice_GetID(device)
                    lengths[i] = len(binary)
                    bins[i] = binary
                    
                self.program_id = clCreateProgramWithBinary(ctx, num_devices, device_list, lengths, bins, binary_status, & err_code)
    
                if err_code != CL_SUCCESS:
                    raise OpenCLException(err_code)
                
                for i in range(num_devices):
                    status = binary_status[i]
                    if status != CL_SUCCESS:
                        raise OpenCLException(status)
            except:
                free(device_list)
                free(lengths)
                free(bins)
                free(binary_status)
                raise
            free(device_list)
            free(lengths)
            free(bins)
            free(binary_status)
            
            
            
    def build(self, devices=None, options='', do_raise=True):
        '''

        Builds (compiles & links) a program executable from the program source or binary for all the 
        devices or a specific device(s) in the OpenCL context associated with program.  
        
        OpenCL allows  program executables to be built using the source or the binary.
        '''
        
        cdef cl_int err_code
        
        str_options = options.encode()
        cdef char * _options = str_options
        
        cdef cl_uint num_devices = 0
        cdef cl_device_id * device_list = NULL
        
        err_code = clBuildProgram(self.program_id, num_devices, device_list, _options, NULL, NULL)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)

        cdef cl_build_status bld_status
        cdef cl_int bld_status_
        if do_raise:
            for device, status in self.status.items():
                bld_status_ = < cl_int > status
                bld_status = < cl_build_status > bld_status_
                if bld_status == CL_BUILD_ERROR:
                    raise BuildError(self.logs[device], self.logs)
        return self
    
    property num_devices:
        'number of devices to build on'
        def __get__(self):
            
            cdef cl_int err_code
            cdef cl_uint value = 0 
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_NUM_DEVICES, sizeof(value), & value, NULL)

            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return value
        
    property _reference_count:
        def __get__(self):
            
            cdef cl_int err_code
            cdef cl_uint value = 0 
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_REFERENCE_COUNT, sizeof(value), & value, NULL)

            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return value
        

    property source:
        'get the source code used to build this program'
        def __get__(self):
            
            cdef cl_int err_code
            cdef char * src = NULL
            cdef size_t src_len = 0 
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_SOURCE, 0, NULL, & src_len)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            if src_len <= 1:
                return None
            
            src = < char *> malloc(src_len + 1)
            
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_SOURCE, src_len, src, NULL)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            src[src_len] = 0
            
            return src.decode('UTF-8')

    property binary_sizes:
        'return a dict of device:binary_size for each device associated with this program'

        def __get__(self):
            
            cdef cl_int err_code
            cdef size_t * sizes = NULL
            cdef size_t slen = 0 
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_BINARY_SIZES, 0, NULL, & slen)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            sizes = < size_t *> malloc(slen)
            
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_BINARY_SIZES, slen, sizes, NULL)
            if err_code != CL_SUCCESS: 
                free(sizes)
                raise OpenCLException(err_code)
            
            size_list = []
            for i in range(slen / sizeof(size_t)):
                size_list.append(sizes[i])
            free(sizes)
            
            return size_list
        
    property binaries:
        '''
        return a dict of {device:bytes} for each device associated with this program
        
        Binaries may be used in a program constructor.
        '''
        def __get__(self):
            
            sizes = self.binary_sizes
            
            cdef size_t param_size = sizeof(char *) * len(sizes)
             
            cdef char ** binaries = < char **> malloc(param_size)
            
            for i, size in enumerate(sizes):
                if size > 0:
                    binaries[i] = < char *> malloc(sizeof(char) * size)
                else:
                    binaries[i] = NULL
                    
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_BINARIES, 0, NULL, & param_size)
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_BINARIES, param_size, binaries, NULL)
            
            if err_code != CL_SUCCESS:
                for i in range(len(sizes)):
                    if binaries[i] != NULL: free(binaries[i])
                free(binaries)
                raise OpenCLException(err_code)
            
            py_binaries = []
            
            for i in range(len(sizes)):
                if binaries[i] == NULL:
                    py_binaries.append(None)
                    continue
                
                binary = bytes(binaries[i][:sizes[i]])
                
                py_binaries.append(binary)
                
                free(binaries[i])
            
            free(binaries)
                
            return dict(zip(self.devices, py_binaries))
            
            
    property status:
        '''
        return a dict of {device:int} for each device associated with this program.
        
        Valid statuses:
        
         * Program.NONE
         * Program.ERROR
         * Program.SUCCESS
         * Program.IN_PROGRESS

        '''
        def __get__(self):
            
            statuses = []
            cdef cl_build_status status
            cdef cl_int err_code
            cdef cl_device_id device_id
            
            for device in self.devices:
                
                device_id = CyDevice_GetID(device)

                err_code = clGetProgramBuildInfo(self.program_id, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), & status, NULL)
                 
                if err_code != CL_SUCCESS: 
                    raise OpenCLException(err_code)
                
                statuses.append(< cl_int > status)
                
            return dict(zip(self.devices, statuses))
                
            
    property logs:
        '''
        get the build logs for each device.
        
        return a dict of {device:str} for each device associated with this program.

        '''
        def __get__(self):
            
            logs = []
            cdef size_t log_len
            cdef char * logstr
            cdef cl_int err_code
            cdef cl_device_id device_id
            
            for device in self.devices:
                
                device_id = CyDevice_GetID(device)

                err_code = clGetProgramBuildInfo (self.program_id, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, & log_len)
                
                if err_code != CL_SUCCESS: raise OpenCLException(err_code)
                
                if log_len == 0:
                    logs.append('')
                    continue
                
                logstr = < char *> malloc(log_len + 1)
                err_code = clGetProgramBuildInfo (self.program_id, device_id, CL_PROGRAM_BUILD_LOG, log_len, logstr, NULL)
                 
                if err_code != CL_SUCCESS: 
                    free(logstr)
                    raise OpenCLException(err_code)
                
                logstr[log_len] = 0
                logs.append(logstr.decode('UTF-8'))
                
            return dict(zip(self.devices, logs))
                
        
    property context:
        'get the context associated with this program'
        def __get__(self):
            
            cdef cl_int err_code
            cdef cl_context ctx = NULL
            
            err_code = clGetProgramInfo(self.program_id, CL_PROGRAM_CONTEXT, sizeof(cl_context), & ctx, NULL)
              
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return CyContext_Create(ctx)
        
    def __getattr__(self, attr):
        return self.kernel(attr)
    
    def kernel(self, name):
        '''
        Return a kernel object. 
        '''
        cdef cl_int err_code
        cdef cl_kernel kernel_id
        str_name = name.encode()
        cdef char * kernel_name = str_name
        
        kernel_id = clCreateKernel(self.program_id, kernel_name, & err_code)
    
        if err_code != CL_SUCCESS:
            if err_code == CL_INVALID_KERNEL_NAME:
                raise KeyError('kernel %s not found in program' % name)
            raise OpenCLException(err_code, clCreateKernel_errors)
        
        return KernelAsPyKernel(kernel_id)

    property devices:
        'returns a list of devices associate with this program.'
        def __get__(self):
            
            cdef cl_int err_code
            cdef cl_device_id * device_list
                        
            cdef cl_uint num_devices = self.num_devices
            
            device_list = < cl_device_id *> malloc(sizeof(cl_device_id) * num_devices)
            err_code = clGetProgramInfo (self.program_id, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * num_devices, device_list, NULL)
            
            if err_code != CL_SUCCESS:
                free(device_list)
                raise OpenCLException(err_code)
            
            
            devices = []
            
            for i in range(num_devices):
                devices.append(CyDevice_Create(device_list[i]))
                
            free(device_list)
            
            return devices
        

cdef api object CyProgram_Create(cl_program program_id):
    cdef Program prog = < Program > Program.__new__(Program)
    prog.program_id = program_id
    clRetainProgram(program_id)
    return prog

