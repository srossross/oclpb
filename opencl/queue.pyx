
from opencl.errors import OpenCLException
import weakref

from _cl cimport * 

from libc.stdlib cimport malloc, free 
from cpython cimport PyObject, Py_DECREF, Py_INCREF, PyBuffer_IsContiguous, PyBuffer_FillContiguousStrides
from cpython cimport Py_buffer, PyBUF_SIMPLE, PyBUF_STRIDES, PyBUF_ND, PyBUF_FORMAT, PyBUF_INDIRECT, PyBUF_WRITABLE

from opencl.copencl cimport CyDevice_GetID, CyDevice_Create 
from opencl.event cimport PyEvent_New, cl_eventFrom_PyEvent, PyEvent_Check 
from opencl.context cimport CyContext_GetID, CyContext_Create, CyContext_Check
from opencl.kernel cimport KernelFromPyKernel
from opencl.cl_mem cimport CyMemoryObject_GetID, CyMemoryObject_Check, CyView_GetPyBuffer, CyView_GetBuffer, CyImage_GetBuffer


cdef extern from "Python.h":
    object PyByteArray_FromStringAndSize(char * , Py_ssize_t)
    object PyMemoryView_FromBuffer(Py_buffer * info)
    int PyObject_GetBuffer(object obj, Py_buffer * view, int flags)
    int PyObject_CheckBuffer(object obj)
    void PyBuffer_Release(Py_buffer * view)

    void PyEval_InitThreads()

MAGIC_NUMBER = 0xabc123
    
PyEval_InitThreads()


cdef class UserData:
    cdef int magic
    cdef object function
    cdef object args
    cdef object kwargs
    cdef void ** args_mem_loc
     
     
cdef void user_func(UserData user_data) with gil:
    
    if user_data.magic != MAGIC_NUMBER:
        raise Exception("Enqueue native kernel can not be used at this time") 

    function = user_data.function
    args = user_data.args
    kwargs = user_data.kwargs
    
    function(*args, **kwargs)
    
    user_data.function = None
    user_data.args = None
    user_data.kwargs = None
    
    Py_DECREF(user_data)
    return
    
_enqueue_copy_buffer_errors = {
                               

    CL_INVALID_COMMAND_QUEUE: 'if command_queue is not a valid command-queue.',

    CL_INVALID_CONTEXT: ('The context associated with command_queue, src_buffer and '
    'dst_buffer are not the same or if the context associated with command_queue and events ' 
    'in event_wait_list are not the same.'),
                               
    CL_INVALID_MEM_OBJECT: 'source and dest are not valid buffer objects.',
    
    CL_INVALID_VALUE : ('source, dest, size, src_offset / cb or dst_offset / cb '
                        'require accessing elements outside the src_buffer and dst_buffer buffer objects ' 
                        'respectively. '),
    CL_INVALID_EVENT_WAIT_LIST :('event_wait_list is NULL and ' 
                                 'num_events_in_wait_list > 0, or event_wait_list is not NULL and ' 
                                 'num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events.'),
                               }

nd_range_kernel_errors = {
    CL_INVALID_PROGRAM_EXECUTABLE : ('There is no successfully built program '
                                     'executable available for device associated with command_queue.'),
    CL_INVALID_COMMAND_QUEUE : 'command_queue is not a valid command-queue.',
    CL_INVALID_KERNEL :'kernel is not a valid kernel object',
    CL_INVALID_CONTEXT: ('Context associated with command_queue and kernel are not ' 
                         'the same or if the context associated with command_queue and events in event_wait_list '
                         'are not the same.'),
    CL_INVALID_KERNEL_ARGS : 'The kernel argument values have not been specified.',
    CL_INVALID_WORK_DIMENSION : 'work_dim is not a valid value',
    CL_INVALID_GLOBAL_WORK_SIZE : ('global_work_size is NULL, or if any of the ' 
                                   'values specified in global_work_size[0], ... ' 
                                   'global_work_size[work_dim - 1] are 0 or ' 
                                   'exceed the range given by the sizeof(size_t) for the device on which the kernel ' 
                                   'execution will be enqueued.'),
    CL_INVALID_GLOBAL_OFFSET : ('The value specified in global_work_size + the ' 
                                'corresponding values in global_work_offset for any dimensions is greater than the ' 
                                'sizeof(size t) for the device on which the kernel execution will be enqueued. '),
    CL_INVALID_WORK_GROUP_SIZE :('local_work_size is specified and number of workitems specified by global_work_size is not evenly divisible by size of work-group given ' 
                                 'by local_work_size or does not match the work-group size specified for kernel'),
    CL_INVALID_WORK_GROUP_SIZE : ('local_work_size is specified and the total number ' 
                                  'of work-items in the work-group computed as local_work_size[0] * ... ' 
                                  'local_work_size[work_dim - 1] is greater than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE'),
    CL_INVALID_WORK_GROUP_SIZE : ('local_work_size is NULL and the ' 
                                  '__attribute__((reqd_work_group_size(X, Y, Z))) qualifier is used to ' 
                                  'declare the work-group size for kernel in the program source. '),
    CL_INVALID_WORK_ITEM_SIZE :('The number of work-items specified in any of ' 
                                'local_work_size[0], ... local_work_size[work_dim - 1]    is greater than the ' 
                                'corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0], ...CL_DEVICE_MAX_WORK_ITEM_SIZES[work_dim - 1].'),
    CL_MISALIGNED_SUB_BUFFER_OFFSET : ('A sub-buffer object is specified as the value ' 
                                       'for an argument that is a buffer object and the offset specified when the sub-buffer object ' 
                                       'is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device ' 
                                       'associated with queue.'),
    CL_INVALID_IMAGE_SIZE : ('An image object is specified as an argument value and the ' 
                             'image dimensions (image width, height, specified or compute row and/or slice pitch) are ' 
                             'not supported by device associated with queue.'),
    CL_OUT_OF_RESOURCES  : 'CL_OUT_OF_RESOURCES, There is a failure to queue the execution instance of kernel ',
    CL_MEM_OBJECT_ALLOCATION_FAILURE :('There is a failure to allocate memory for ' 
                                       'data store associated with image or buffer objects specified as arguments to kernel. '),
    CL_INVALID_EVENT_WAIT_LIST: ('event_wait_list is NULL and ' 
                                 'num_events_in_wait_list > 0, or event_wait_list is not NULL and ' 
                                 'num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events. '),
    CL_OUT_OF_HOST_MEMORY : ('There is a failure to allocate resources required by the' 
                             'OpenCL implementation on the host')
}

cdef class Queue:
    '''
    opencl.Queue(context, device=None, out_of_order_exec_mode=False, profiling=False)
    
    OpenCL objects such as memory, program and kernel objects are created using a context.  
    Operations on these objects are performed using  a command-queue. The command-queue can be 
    used to queue a set of operations (referred to as commands) in order.  Having multiple 
    command-queues allows applications to queue multiple independent commands without 
    requiring synchronization.  Note that this should work as long as these objects are not being 
    shared.  Sharing of objects across multiple command-queues will require the application to 
    perform appropriate synchronization
    
    :param context: An opencl.Context object 
    :param device: if None use the first device in the context [default None] 
    :param out_of_order_exec_mode: enable out_of_order_exec_mode [default False] 
    :param profiling: enable profiling [default False]
    '''
    cdef cl_command_queue queue_id
    
    def __cinit__(self):
        self.queue_id = NULL
    
    def __dealloc__(self):
        if self.queue_id != NULL:
            clReleaseCommandQueue(self.queue_id)
        self.queue_id = NULL
    
    def __init__(self, context, device=None, out_of_order_exec_mode=False, profiling=False):
        
        if not CyContext_Check(context):
            raise TypeError("argument 'context' must be a valid opencl.context object (got %s)" % type(context))
            
        if device is None:
            if context.num_devices != 1:
                raise TypeError("must specify a device. context does does not contain a unique device (has %i)" % (context.num_devices))
            device = context.devices[0]
            
        cdef cl_command_queue_properties properties = 0
        
        properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE if out_of_order_exec_mode else 0
        properties |= CL_QUEUE_PROFILING_ENABLE if profiling else 0
            
        cdef cl_int err_code = CL_SUCCESS
       
        cdef cl_context ctx = CyContext_GetID(context)
        cdef cl_device_id device_id = CyDevice_GetID(device)
         
        self.queue_id = clCreateCommandQueue(ctx, device_id, properties, & err_code)
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)

    property device:
        '''
        Return the device associated with this queue
        '''
        def __get__(self):
            cdef cl_int err_code
            cdef cl_device_id device_id
             
            err_code = clGetCommandQueueInfo (self.queue_id, CL_QUEUE_DEVICE, sizeof(cl_device_id), & device_id, NULL)
            
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return CyDevice_Create(device_id) 

    property context:
        '''
        Return the context that this queue was created with
        '''

        def __get__(self):
            cdef cl_int err_code
            cdef cl_context context_id
             
            err_code = clGetCommandQueueInfo (self.queue_id, CL_QUEUE_CONTEXT, sizeof(cl_context), & context_id, NULL)
            
            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return CyContext_Create(context_id) 
        
    def barrier(self):
        '''
        queue.barrier()
        
        Enqueues a barrier operation.
        The queue.barrier command ensures that all queued 
        commands in command_queue have finished execution before the next batch of commands can 
        begin execution.  The queue.barrier command is a synchronization point
        '''
        cdef cl_int err_code
        cdef cl_command_queue queue_id = self.queue_id
         
        with nogil:
            err_code = clEnqueueBarrier(queue_id)

        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
    def flush(self):
        '''
        queue.flush()
        
        Issues all previously queued OpenCL commands in command_queue to the device associated 
        with command_queue.  clFlush only guarantees that all queued commands to command_queue
        will eventually be submitted to the appropriate device.  There is no guarantee that they will be 
        complete after clFlush returns. 
        '''
        
        cdef cl_int err_code
         
        err_code = clFlush(self.queue_id)

        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)

    def finish(self):
        '''
        queue.finish()
        
        Blocks until all previously queued OpenCL commands in command_queue are issued to the 
        associated device and have completed.  clFinish does not return until all queued commands in 
        command_queue have been processed and completed.  clFinish is also a synchronization point
        
        '''
        cdef cl_int err_code
        cdef cl_command_queue queue_id = self.queue_id
        
        with nogil:
            err_code = clFinish(queue_id) 

        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
    def marker(self):
        '''queue.marker()
        
        Enqueues a marker command to command_queue.  The marker command is not completed until 
        all commands enqueued before it have completed.  The marker command returns an event which 
        can be waited on, i.e. this event can be waited on to insure that all commands, which have been 
        queued before the marker command, have been completed. 
        '''
        
        cdef cl_event event_id = NULL
        cdef cl_int err_code
         
        err_code = clEnqueueMarker(self.queue_id, & event_id)
         
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        return PyEvent_New(event_id)
        
    def copy(self, source, dest):
        pass
    
    def enqueue_wait_for_events(self, *events):
        '''
        queue.enqueue_wait_for_events(self, event, event2, ...)
        queue.enqueue_wait_for_events(self, eventlist)
        
        Enqueues a wait for a specific event or a list of events to complete before any future commands 
        queued in the command-queue are executed.  num_events specifies the number of events given 
        by event_list. 
        
        '''
        if len(events) == 1:
            if isinstance(events[0], (list, tuple)):
                events = events[0]
            else:
                events = (events[0],)
        
        cdef cl_event * event_wait_list
        cdef cl_uint num_events_in_wait_list = _make_wait_list(events, & event_wait_list)
        
        if event_wait_list == < cl_event *> 1:
            raise Exception("One of the items in argument 'wait_on' is not a valid event")
        
        cdef cl_uint err_code
        
        err_code = clEnqueueWaitForEvents(self.queue_id, num_events_in_wait_list, event_wait_list)

        free(event_wait_list)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
    
    def enqueue_native_kernel(self, function, *args, **kwargs):
        '''
        queue.enqueue_native_kernel(function [, arg, ..., kwarg=, ...])
        
        Enqueues a command to execute a python function.
        
        :param function: A callable python object
        :param args: Arguments for function
        :param kwargs: Keywords for function
        
        '''
        cdef UserData user_data = UserData() 
        
        user_data.magic = MAGIC_NUMBER 
        
        user_data.function = function
        user_data.args = args
        user_data.kwargs = kwargs
        
        cdef cl_mem * mem_list = NULL

        cdef int nbuffers = 0
        user_data.args_mem_loc = NULL
                
        Py_INCREF(user_data)
                    
        cdef cl_int err_code
        cdef cl_event event_id
        cdef cl_uint num_events_in_wait_list = 0
        cdef cl_event * event_wait_list = NULL

        cdef void * _args = < void *> user_data
        cdef size_t cb_args = sizeof(UserData)
        
        err_code = clEnqueueNativeKernel(self.queue_id,
                                      < void *>& user_func,
                                      _args,
                                      cb_args,
                                      nbuffers,
                                      mem_list,
                                      user_data.args_mem_loc,
                                      num_events_in_wait_list,
                                      event_wait_list,
                                      & event_id) 
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        return PyEvent_New(event_id)
    
    
    def enqueue_task(self, kernel, wait_on=()):
        '''queue.enqueue_task(kernel,  wait_on=())
        
        Enqueues a command to execute a kernel on a device.  The kernel is executed using a single 
        work-item.
        
        :param kernel: an opencl kernel.
        :param wait_on: a list of events
        '''
        
        cdef cl_event * event_wait_list
        cdef cl_uint num_events_in_wait_list = _make_wait_list(wait_on, & event_wait_list)
        cdef cl_int err_code

        cdef cl_kernel kernel_id = KernelFromPyKernel(kernel)
        cdef cl_event event_id = NULL

        err_code = clEnqueueTask(self.queue_id, kernel_id, num_events_in_wait_list, event_wait_list, & event_id)

        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code, nd_range_kernel_errors)
        
        return PyEvent_New(event_id)


    def enqueue_nd_range_kernel(self, kernel, cl_uint  work_dim,
                                global_work_size, global_work_offset=None, local_work_size=None, wait_on=()):
        '''queue.enqueue_nd_range_kernel(kernel, work_dim, global_work_size, global_work_offset=None, local_work_size=None, wait_on=())
        
        Enqueues a command to execute a kernel on a device
        
        :param kernel: an opencl.Kernel object 
        :param work_dim:  is the number of dimensions used to specify the global work-items  and work-items in the work-group.
        :param global_work_size: A list of length work_dim that describe the number of global 
                work-items in work_dim dimensions that will execute the kernel function.  T
        :param global_work_offset: Can be used to specify an array of work_dim unsigned values that describe 
                the offset used to calculate the global ID of a work-item.
        :param local_work_size:f A list of length work_dim unsigned values that describe the number of 
                work-items that make up a work-group.
                If None, the OpenCL implementation will determine how to be break the global work-items into 
                appropriate work-group instances
        :param wait_on: A list of events
        
        '''
        
        cdef cl_kernel kernel_id = KernelFromPyKernel(kernel)
        
        cdef cl_event event_id = NULL
        
        cdef size_t * gsize = < size_t *> malloc(sizeof(size_t) * work_dim)
        cdef size_t * goffset = NULL
        cdef size_t * lsize = NULL
        if global_work_offset:
            if len(global_work_offset) != len(global_work_size):
                raise TypeError('dimentionality of global_work_offset (%r dims) does not match global_work_size (%r dims)' % (len(global_work_offset), len(global_work_size)))
            goffset = < size_t *> malloc(sizeof(size_t) * work_dim)
        if local_work_size:
            if len(local_work_size) != len(global_work_size):
                free(goffset)
                raise TypeError('dimentionality of local_work_size (%r dims) does not match global_work_size (%r dims)' % (len(local_work_size), len(global_work_size)))
                
            lsize = < size_t *> malloc(sizeof(size_t) * work_dim)
        
        for i in range(work_dim):
            gsize[i] = < size_t > global_work_size[i]
            if goffset != NULL: goffset[i] = < size_t > global_work_offset[i]
            if lsize != NULL: lsize[i] = < size_t > local_work_size[i]
            
        cdef cl_event * event_wait_list
        cdef cl_uint num_events_in_wait_list = _make_wait_list(wait_on, & event_wait_list)
        cdef cl_int err_code

        err_code = clEnqueueNDRangeKernel(self.queue_id, kernel_id,
                                          work_dim, goffset, gsize, lsize,
                                          num_events_in_wait_list, event_wait_list, & event_id)
        
        if gsize != NULL: free(gsize)
        if goffset != NULL: free(goffset)
        if lsize != NULL: free(lsize)

        if err_code != CL_SUCCESS:
            if err_code == CL_INVALID_WORK_GROUP_SIZE:

                if any([(x % y) != 0 for x, y in zip(global_work_size, local_work_size)]):
                    msg = 'local work size %s does not divide global_work_size %r evenly' % (local_work_size, global_work_size)
                    raise OpenCLException(err_code, msg=msg)
                
                work_group_size = kernel.work_group_size(self.device)
                prod_shape = 1
                for d in local_work_size:
                    prod_shape *= d
                if work_group_size < prod_shape:
                    ps = '*'.join([str(x) for x in local_work_size])
                    msg = 'total workgroup size (%s) excceds maximum defined by "kernel.work_group_size(queue.device)" of %r' % (ps, work_group_size)
                    raise OpenCLException(err_code, msg=msg)
                
                if any([x > y for x, y in zip(local_work_size, self.device.max_work_item_sizes)]):
                    msg = 'an item in local work size (%s) excceds maximum defined "device.max_work_item_sizes" %r' % (local_work_size, self.device.max_work_item_sizes)
                    raise OpenCLException(err_code, msg=msg)

                    
            raise OpenCLException(err_code, nd_range_kernel_errors)
        
        return PyEvent_New(event_id)
    
    def enqueue_copy_buffer(self, source, dest, size_t src_offset=0, size_t dst_offset=0, size_t size=0, wait_on=()):
        '''
        queue.enqueue_copy_buffer(source, dest, src_offset=0, dst_offset=0, size=0, wait_on=())
                
        Enqueues a command to copy a buffer object identified by source to another buffer object 
        identified by dest.
        
        :param source: memory object
        :param dest:  memory object
        :param src_offset: refers to the offset where to begin copying data from src
        :param dst_offset: refers to the offset where to begin copying data into dest
        :param size: number of bytes to copy
        :param wait_on: a sequence of events to wait for before submitting this command. 
        '''
        cdef cl_int err_code
        cdef cl_event event_id = NULL
        cdef cl_event * event_wait_list
        cdef cl_uint num_events_in_wait_list = _make_wait_list(wait_on, & event_wait_list)
        
        if not CyMemoryObject_Check(source):
            raise TypeError("Argument 'source' must be a valid memory object")
        if not CyMemoryObject_Check(dest):
            raise TypeError("Argument 'dest' must be a valid memory object")
        
        cdef cl_mem src_buffer = CyMemoryObject_GetID(source)
        cdef cl_mem dst_buffer = CyMemoryObject_GetID(dest)
        
        err_code = clEnqueueCopyBuffer(self.queue_id, src_buffer, dst_buffer, src_offset, dst_offset, size,
                                       num_events_in_wait_list, event_wait_list, & event_id)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code, _enqueue_copy_buffer_errors)
    
        return PyEvent_New(event_id)

    def enqueue_read_buffer(self, source, dest, size_t src_offset=0, size_t size=0, wait_on=(), cl_bool blocking_read=0):
        '''
        queue.enqueue_read_buffer(source, dest, offset=0, size=0, wait_on=(), blocking_read=False)
        
        Read a buffer object to host memory.
        '''
        cdef cl_int err_code
        cdef cl_event event_id = NULL
        cdef cl_event * event_wait_list
        cdef cl_uint num_events_in_wait_list = _make_wait_list(wait_on, & event_wait_list)
        
        cdef cl_mem src_buffer = CyMemoryObject_GetID(source)
        
        cdef int flags = PyBUF_SIMPLE
        
        if not PyObject_CheckBuffer(dest):
            raise Exception("dest argument of enqueue_read_buffer is required to be a new style buffer object (got %r)" % dest)

        cdef Py_buffer dst_buffer
        
        if PyObject_GetBuffer(dest, & dst_buffer, flags) < 0:
            raise Exception("dest argument of enqueue_read_buffer is required to be a new style buffer object")
        
        if dst_buffer.len < size:
            raise Exception("dest buffer must be at least `size` bytes")
        
        if not PyBuffer_IsContiguous(& dst_buffer, 'A'):
            raise Exception("dest buffer must be contiguous")
        
        err_code = clEnqueueReadBuffer(self.queue_id, src_buffer, blocking_read, src_offset, size, dst_buffer.buf,
                                       num_events_in_wait_list, event_wait_list, & event_id)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code, _enqueue_copy_buffer_errors)
    
        return PyEvent_New(event_id)
    
    def enqueue_write_buffer(self, source, dest, size_t src_offset=0, size_t size=0, wait_on=(), cl_bool blocking_read=0):
        '''
        queue.enqueue_read_buffer(source, dest, offset=0, size=0, wait_on=(), blocking_read=False)
        
        Write host memory into a buffer object.
        
        
        '''
        cdef cl_int err_code
        cdef cl_event event_id = NULL
        cdef cl_event * event_wait_list
        cdef cl_uint num_events_in_wait_list = _make_wait_list(wait_on, & event_wait_list)
        
        cdef cl_mem src_buffer = CyMemoryObject_GetID(source)
        
        cdef int flags = PyBUF_SIMPLE | PyBUF_WRITABLE
        
        if not PyObject_CheckBuffer(dest):
            raise Exception("dest argument of enqueue_read_buffer is required to be a new style buffer object (got %r)" % dest)

        cdef Py_buffer dst_buffer
        
        if PyObject_GetBuffer(dest, & dst_buffer, flags) < 0:
            raise Exception("dest argument of enqueue_read_buffer is required to be a new style buffer object")
        
        if dst_buffer.len < size:
            raise Exception("dest buffer must be at least `size` bytes")
        
        if not PyBuffer_IsContiguous(& dst_buffer, 'A'):
            raise Exception("dest buffer must be contiguous")

        if dst_buffer.readonly:
            raise Exception("host buffer must have write access")
        
        err_code = clEnqueueWriteBuffer(self.queue_id, src_buffer, blocking_read, src_offset, size, dst_buffer.buf,
                                       num_events_in_wait_list, event_wait_list, & event_id)
        
        PyBuffer_Release(& dst_buffer)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code, _enqueue_copy_buffer_errors)
    
        return PyEvent_New(event_id)


    def enqueue_copy_buffer_rect(self, source, dest, region, src_origin=(0, 0, 0), dst_origin=(0, 0, 0),
                                 size_t src_row_pitch=0, size_t src_slice_pitch=0,
                                 size_t dst_row_pitch=0, size_t dst_slice_pitch=0, wait_on=()):
        '''
        queue.enqueue_copy_buffer_rect(source, dest, region, src_origin=(0, 0, 0), dst_origin=(0, 0, 0),
                                       src_row_pitch=0, src_slice_pitch=0,
                                       dst_row_pitch=0, dst_slice_pitch=0, wait_on=())
                                       
        TODO: document this.
        '''
        
        cdef cl_int err_code
        cdef cl_event event_id = NULL
        cdef cl_event * event_wait_list
        
        cdef cl_uint num_events_in_wait_list = _make_wait_list(wait_on, & event_wait_list)
        
        cdef cl_mem src_buffer = CyMemoryObject_GetID(source)
        
        cdef cl_mem dst_buffer = CyMemoryObject_GetID(dest)
        
        cdef size_t _src_origin[3]
        _src_origin[:] = [0, 0, 0]
        cdef  size_t _dst_origin[3]
        _dst_origin[:] = [0, 0, 0]
        cdef size_t _region[3]
        _region[:] = [1, 1, 1]
        
        for i, origin in enumerate(src_origin):
            _src_origin[i] = origin

        for i, origin in enumerate(dst_origin):
            _dst_origin[i] = origin

        for i, size in enumerate(region):
            _region[i] = size

        err_code = clEnqueueCopyBufferRect(self.queue_id, src_buffer, dst_buffer,
                                           _src_origin, _dst_origin, _region,
                                           src_row_pitch, src_slice_pitch,
                                           dst_row_pitch, dst_slice_pitch,
                                           num_events_in_wait_list, event_wait_list, & event_id)
                
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code, _enqueue_copy_buffer_errors)
        
        event = PyEvent_New(event_id)
        return event
    
    def __richcmp__(Queue self, other, op):
        
        if not isinstance(other, Queue):
            return NotImplemented
        
        if op == 2:
            return self.queue_id == CyQueue_GetID(other)
        else:
            return NotImplemented

cdef class MemoryViewMap:
    '''
    context manager for mapping and unmapping buffers.
    '''
    
    cdef cl_command_queue command_queue
    cdef public object dview
    
    cdef cl_bool blocking_map
    cdef cl_map_flags map_flags
    cdef size_t offset
    cdef size_t cb
    cdef void * bytes 
        
    def __init__(self, queue, dview, cl_bool blocking_map, cl_map_flags map_flags):
        
        self.dview = weakref.ref(dview)
        
        self.command_queue = CyQueue_GetID(queue)
        
        self.blocking_map = blocking_map
        self.map_flags = map_flags
    
    def __enter__(self):
        cdef void * bytes 
        cdef cl_uint num_events_in_wait_list = 0
        cdef cl_event * event_wait_list = NULL
        
        cdef cl_int err_code
        
        cdef cl_mem memobj = CyMemoryObject_GetID(self.dview())
        cdef size_t mem_size = self.dview().mem_size
        with nogil:
            bytes = clEnqueueMapBuffer(self.command_queue, memobj,
                                       self.blocking_map, self.map_flags, 0, mem_size,
                                       num_events_in_wait_list, event_wait_list, NULL,
                                       & err_code)
         
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        self.bytes = bytes
        return memoryview(self)

    def __exit__(self, *args):

        cdef cl_int err_code 
        
        cdef cl_mem memobj = CyMemoryObject_GetID(self.dview())
        
        err_code = clEnqueueUnmapMemObject(self.command_queue, memobj, self.bytes, 0, NULL, NULL)
        
        clEnqueueBarrier(self.command_queue)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        
    def __getbuffer__(self, Py_buffer * view, int flags):
        view.len = self.cb
        
        cdef Py_buffer buffer
        
        CyView_GetBuffer(self.dview(), & buffer)
        
        cdef cl_mem memobj = CyMemoryObject_GetID(self.dview())
        writable = bool(self.map_flags & CL_MAP_WRITE)

        view.readonly = 0 if writable else 1 
        

        
        view.format = buffer.format
        view.ndim = buffer.ndim
        view.shape = buffer.shape
        view.itemsize = buffer.itemsize
        view.internal = NULL
        view.strides = buffer.strides
        view.suboffsets = NULL
        
        view.buf = self.bytes + <size_t>buffer.buf
        
    def __releasebuffer__(self, Py_buffer * view):
        pass
    
cdef class ImageMap:
    
    cdef cl_command_queue command_queue
    cdef public object dview
    
    cdef cl_bool blocking_map
    cdef cl_map_flags map_flags
    cdef size_t offset
    cdef size_t image_row_pitch
    cdef size_t image_slice_pitch
    cdef void * bytes
        
    def __init__(self, queue, dview, cl_bool blocking_map, cl_map_flags map_flags):
        
        self.dview = weakref.ref(dview)
        
        self.command_queue = CyQueue_GetID(queue)
        
        self.blocking_map = blocking_map
        self.map_flags = map_flags
    
    def __enter__(self):
        cdef void * bytes 
        cdef cl_uint num_events_in_wait_list = 0
        cdef cl_event * event_wait_list = NULL
        
        cdef cl_int err_code
        
        cdef cl_mem memobj = CyMemoryObject_GetID(self.dview())
        cdef size_t origin[3]
        cdef size_t region[3]
        cdef size_t image_row_pitch
        cdef size_t image_slice_pitch

        cdef Py_buffer buffer
        CyImage_GetBuffer(self.dview(), & buffer)
        
        origin[0] = 0
        origin[1] = 0
        origin[2] = 0
            
        region[0] = buffer.shape[0]
        region[1] = buffer.shape[1]
        region[2] = 1
        
        if buffer.ndim == 3:
            region[2] = buffer.shape[2]
        
        bytes = clEnqueueMapImage(self.command_queue, memobj, self.blocking_map, self.map_flags,
                                  origin, region, & image_row_pitch, & image_slice_pitch,
                                  num_events_in_wait_list, event_wait_list, NULL,
                                  & err_code)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        self.bytes = bytes
        
        self.image_row_pitch = image_row_pitch
        self.image_slice_pitch = image_slice_pitch

        return memoryview(self)
    

    def __exit__(self, *args):

        cdef cl_int err_code
        
        cdef cl_mem memobj = CyMemoryObject_GetID(self.dview())
        
        err_code = clEnqueueUnmapMemObject(self.command_queue, memobj, self.bytes, 0, NULL, NULL)
        clEnqueueBarrier(self.command_queue)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
    def __getbuffer__(self, Py_buffer * view, int flags):
        cdef Py_buffer buffer
        
        CyImage_GetBuffer(self.dview(), & buffer)
        
        writable = bool(self.map_flags & CL_MAP_WRITE)

        view.readonly = 0 if writable else 1 
        
        view.format = buffer.format
        view.ndim = buffer.ndim
        view.shape = buffer.shape
        view.itemsize = buffer.itemsize
        view.internal = NULL
        view.strides = buffer.strides
        view.suboffsets = NULL
        
        view.buf = self.bytes
        
    def __releasebuffer__(self, Py_buffer * view):
        pass

cdef api cl_uint _make_wait_list(wait_on, cl_event ** event_wait_list_ptr):
    if not wait_on:
        event_wait_list_ptr[0] = NULL
        return 0
    cdef cl_uint num_events = len(wait_on)
    cdef cl_event event_id = NULL
    
    cdef cl_event * event_wait_list = < cl_event *> malloc(sizeof(cl_event) * num_events)
    
    for i, pyevent in enumerate(wait_on):
        if not PyEvent_Check(pyevent):
            event_wait_list_ptr[0] = < cl_event *> 1
            return 0
        event_id = cl_eventFrom_PyEvent(pyevent)
        event_wait_list[i] = event_id
        
    event_wait_list_ptr[0] = event_wait_list
    
    return num_events
    

cdef api int CyQueue_Check(object queue):
    return isinstance(queue, Queue)

cdef api cl_command_queue CyQueue_GetID(object queue):
    return (< Queue > queue).queue_id

cdef api object CyQueue_Create(cl_command_queue queue_id):
    cdef Queue queue = < Queue > Queue.__new__(Queue)
    clRetainCommandQueue(queue_id)
    queue.queue_id = queue_id
    return queue
