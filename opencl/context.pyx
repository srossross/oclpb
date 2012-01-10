from opencl.errors import OpenCLException

from _cl cimport * 
 
from opencl.copencl cimport CyPlatform_GetID, CyPlatform_Create
from opencl.copencl cimport CyDevice_GetID, CyDevice_Create

from libc.stdlib cimport malloc, free 

from cpython cimport Py_INCREF 

cdef class ContextProperties:
    '''
    Store of key value pairs that can be used to initialize
    an opencl.Context object.
    '''
    
    cdef public object properties_dict
    property_names_lookup = { < size_t > CL_CONTEXT_PLATFORM : 'platform'}
    
    def __cinit__(self):
        self.properties_dict = {}
        
    property platform:
        def __get__(self):
            cdef cl_platform_id platform_id = NULL
            
            if 'platform' in self.properties_dict:
                plat = < size_t > self.properties_dict['platform'][1]
                platform_id = < cl_platform_id > plat
                return CyPlatform_Create(< cl_platform_id > plat)
            else:
                return None

        def __set__(self, value):
            cdef cl_platform_id platform_id = CyPlatform_GetID(value)
            self.properties_dict['platform'] = (< size_t > CL_CONTEXT_PLATFORM, < size_t > platform_id)

    def set_property(self, name, size_t property, size_t value):
        self.properties_dict[name] = (property, value)
    
    def as_dict(self):
        return self.properties_dict
    
    cdef cl_context_properties * context_properties(self):
        
        nprops = len(self.properties_dict)
        
        cdef cl_context_properties * props = NULL
            
        if nprops == 0:
            return NULL
            
        props = < cl_context_properties *> malloc(sizeof(cl_context_properties) * (1 + 2 * nprops))

        cdef size_t property
        cdef size_t value
        cdef int i
        for i, (property, value) in enumerate(self.properties_dict.values()):
            props[i * 2] = < cl_context_properties > property
            props[i * 2 + 1] = < cl_context_properties > value

        props[nprops * 2] = < cl_context_properties > NULL
        
        return props
    
    def __repr__(self):
#        items = ['%s=%r' % (key, value[1]) for (key, value) in self.properties_dict.items()]
        d = self.properties_dict
        return '<ContextProperties %r>' % (d,)


_context_errors = {
                       CL_INVALID_PLATFORM : ('Properties is NULL and no platform could be selected or if ' 
                                              'platform value specified in properties is not a valid platform.'),
                   CL_INVALID_PROPERTY: ('Context property name in properties is not a supported ' 
                                         'property name, if the value specified for a supported property name is not valid, or if the ' 
                                         'same property name is specified more than once.'),
                   CL_INVALID_VALUE: 'pfn_notify is NULL but user_data is not NULL.',
                   CL_INVALID_DEVICE_TYPE :'device_type is not a valid value.',
                   CL_DEVICE_NOT_AVAILABLE : ('No devices that match device_type and property values' 
                                              'specified in properties are currently available.'),
                                               
                   CL_DEVICE_NOT_FOUND: ('No devices that match device_type and property values ' 
                                        'specified in properties were found.'),
                   CL_OUT_OF_RESOURCES: ('There is a failure to allocate resources required by the ' 
                                         'OpenCL implementation on the device.'),
                   CL_OUT_OF_HOST_MEMORY :('There is a failure to allocate resources required by the ' 
                                           'OpenCL implementation on the host'),
                   }

cdef void pfn_context_err_notify(char * errinfo, void * private_info, size_t cb, object user_data) with gil:

    cdef str info = errinfo
    cdef bytes pr_info = (< char *> private_info)[:cb]
    
    user_data(info, pr_info)

def print_context_error(info, private_info):
    print "OpenCL context error: %s" % (info)
    
cdef class Context:
    '''
    opencl.Context(devices=(), device_type=cl.Device.DEFAULT, ContextProperties properties=None, callback=print_context_error)
    
    
    Creates an OpenCL context. An OpenCL context is created with one or more devices.  Contexts 
    are used by the OpenCL runtime for managing objects such as command-queues, memory, 
    program and kernel objects and for executing kernels on one or more devices specified in the 
    context.
    
    :param devices: list of opencl devices
    :param device_type: type of device to create context from. used only if `devices` is empty.  
    :param properties: cl.ContextProperties object 
    :param callback: This callback function 
        will be used by the OpenCL implementation to report  information on errors that occur in this 
        context. the function signature must be callback(str, bytes)
    
    '''
    cdef cl_context context_id
    
    def __cinit__(self):
        self.context_id = NULL
        
    def __dealloc__(self):
        if self.context_id != NULL:
            clReleaseContext(self.context_id)
        self.context_id = NULL

        
    def __init__(self, devices=(), device_type=CL_DEVICE_TYPE_DEFAULT, ContextProperties properties=None, callback=None):
        
        cdef cl_context_properties * props = NULL
        
        if properties is not None:
            props = properties.context_properties()
        
        cdef cl_device_type dtype
        cdef cl_int err_code
        cdef cl_uint num_devices
        cdef cl_device_id * _devices = NULL
        
        cdef void * pfn_notify = NULL
        cdef void * user_data = NULL
        
        if callback is not None:
            pfn_notify = < void *> & pfn_context_err_notify
            Py_INCREF(callback) 
            user_data = < void *> callback  
            
        if devices:
            num_devices = len(devices)
            _devices = < cl_device_id *> malloc(num_devices * sizeof(cl_device_id))
            for i in range(num_devices): 
                _devices[i] = CyDevice_GetID(devices[i])
                 
            self.context_id = clCreateContext(props, num_devices, _devices, pfn_notify, user_data, & err_code)
            
            if _devices != NULL:
                free(_devices)

            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code, _context_errors)
        else:
            if isinstance(device_type, (str, bytes)):
                if isinstance(device_type, bytes):
                    device_type = device_type.decode('utf-8')
                import opencl
                
                dtype = < cl_device_type > opencl.Device.DEV_TYPE_MAP[device_type.upper()]
                
            else:  
                dtype = < cl_device_type > device_type
            
            if props == NULL:
                import opencl
                properties = ContextProperties()
                properties.platform = opencl.get_platforms()[0] 
                props = properties.context_properties()
            self.context_id = clCreateContextFromType(props, dtype, pfn_notify, user_data, & err_code)
    
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code, _context_errors)
            
#        TODO: Free props
#        if props != NULL:
#            free(props)
#            props = NULL

    def __repr__(self):
        return '<%s num_devices=%i>' % (type(self).__name__, self.num_devices)
    
    property ref_count:
        'return opencl internal refrence count of this object'
        def __get__(self):
            pass
        
    property num_devices:
        'return the number of devices'
        def __get__(self):
            
            cdef cl_int err_code
            cdef cl_uint num_devices
            err_code = clGetContextInfo (self.context_id, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), & num_devices, NULL)
            
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            return num_devices
            
    property properties:
        'return a ContextProperties object'
        def __get__(self):
            properties = ContextProperties()
            
            cdef cl_int err_code
            cdef size_t props_size
            cdef cl_context_properties * props = NULL
            
            err_code = clGetContextInfo (self.context_id, CL_CONTEXT_PROPERTIES, 0, NULL, & props_size)
            if err_code != CL_SUCCESS: raise OpenCLException(err_code)
            
            if not props_size:
                return properties
            
            props = < cl_context_properties *> malloc(props_size)
            
            err_code = clGetContextInfo (self.context_id, CL_CONTEXT_PROPERTIES, props_size, props, NULL)
            if err_code != CL_SUCCESS:
                free(props) 
                raise OpenCLException(err_code)
            
            if props[0] != NULL:
                
                nprops = props_size // (2 * sizeof(cl_context_properties))
                
                for i in range(nprops):
                    if (< size_t > props[i * 2]) == 0:
                        break
                    else:
                        name = < size_t > props[i * 2]
                        rname = ContextProperties.property_names_lookup.get(name, name)
                        properties.set_property(rname, name, < size_t > props[i * 2 + 1])

                    
            free(props)
            return properties
            
        
    property devices:
        'return a list of devices associated with this context'
        
        def __get__(self):
            
            cdef cl_int err_code
            cdef size_t num_devices = self.num_devices
            cdef cl_device_id * _devices
    
            _devices = < cl_device_id *> malloc(num_devices * sizeof(cl_device_id))
    
            err_code = clGetContextInfo(self.context_id, CL_CONTEXT_DEVICES, num_devices * sizeof(cl_device_id), _devices, NULL)
    
            if err_code != CL_SUCCESS:
                free(_devices)
                raise OpenCLException(err_code)
            
            devices = []
            for i in range(num_devices): 
                device_id = _devices[i]
                device = CyDevice_Create(device_id)
                devices.append(device) 
                
            free(_devices)
            
            return devices
    
    def __hash__(self):
        return < size_t > self.context_id

    def __richcmp__(Context self, other, op):
        
        if not isinstance(other, Context):
            return NotImplemented
        
        if op == 2:
            return self.context_id == CyContext_GetID(other)
        else:
            return NotImplemented
    
#===============================================================================
# API
#===============================================================================

cdef api cl_context CyContext_GetID(object pycontext):
    cdef Context context = < Context > pycontext
    return context.context_id

cdef api object CyContext_Create(cl_context context):
    ctx = < Context > Context.__new__(Context)
    clRetainContext(context)
    ctx.context_id = context
    return ctx

cdef api int CyContext_Check(object context):
    return isinstance(context, Context) 
