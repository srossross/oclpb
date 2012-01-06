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

from opencl.context cimport CyContext_GetID, CyContext_Create, CyContext_Check

cdef extern from "Python.h":
    void PyEval_InitThreads()


cdef void pfn_event_notify(cl_event event, cl_int event_command_exec_status, void * data) with gil:
    
    cdef object user_data = (< object > data)
    
    pyevent = cl_eventAs_PyEvent(event)
    
    try:
        user_data(pyevent, event_command_exec_status)
    except:
        Py_DECREF(< object > user_data)
        raise
    else:
        Py_DECREF(< object > user_data)
    

cdef class Event:
    '''
    An event object can be used to track the execution status of a command.  The API calls that 
    enqueue commands to a command-queue create a new event object that is returned in the event
    argument.
    '''
    QUEUED = CL_QUEUED
    SUBMITTED = CL_SUBMITTED
    RUNNING = CL_RUNNING
    COMPLETE = CL_COMPLETE
    
    STATUS_DICT = { CL_QUEUED: 'queued', CL_SUBMITTED:'submitted', CL_RUNNING: 'running', CL_COMPLETE:'complete'}
    
    cdef cl_event event_id
    
    def __cinit__(self):
        self.event_id = NULL

    def __dealloc__(self):
        if self.event_id != NULL:
            clReleaseEvent(self.event_id)
        self.event_id = NULL
        
    def __repr__(self):
        status = self.status
        return '<%s status=%r:%r>' % (self.__class__.__name__, status, self.STATUS_DICT[status])
    
    def wait(self):
        '''
        event.wait()
        
        Waits on the host thread for commands identified by event objects in event_list to complete.  
        A command is considered complete if its execution status is CL_COMPLETE or a negative value.  
        
        '''
        cdef cl_int err_code
        
        with nogil:
            err_code = clWaitForEvents(1, & self.event_id)
    
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
    property status:
        '''
        the current status of the event.
        '''
        def __get__(self):
            cdef cl_int err_code
            cdef cl_int status

            err_code = clGetEventInfo(self.event_id, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), & status, NULL)

            if err_code != CL_SUCCESS:
                raise OpenCLException(err_code)
            
            return status
        
    property profile_start:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_ulong status

            err_code = clGetEventProfilingInfo(self.event_id, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), & status, NULL)

            if err_code != CL_SUCCESS:
                if err_code == CL_PROFILING_INFO_NOT_AVAILABLE:
                    if self.status != self.COMPLETE:
                        raise OpenCLException(err_code, msg='Event must have (status == Event.COMPLETE) before it can be profiled')
                    else:
                        raise OpenCLException(err_code, msg='Queue must be created with profile=True argument in constructor.')
                
                                
                raise OpenCLException(err_code)
            
            return status
        
    property profile_end:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_ulong status

            err_code = clGetEventProfilingInfo(self.event_id, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), & status, NULL)

            if err_code != CL_SUCCESS:
                if err_code == CL_PROFILING_INFO_NOT_AVAILABLE:
                    if self.status != self.COMPLETE:
                        raise OpenCLException(err_code, msg='Event must have (status == Event.COMPLETE) before it can be profiled')
                    else:
                        raise OpenCLException(err_code, msg='Queue must be created with profile=True argument in constructor.')
                
                                
                raise OpenCLException(err_code)
            
            return status

    property duration:
        def __get__(self):
            return self.profile_end - self.profile_start
    
    property profile_queued:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_ulong status

            err_code = clGetEventProfilingInfo(self.event_id, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), & status, NULL)

            if err_code != CL_SUCCESS:
                if err_code == CL_PROFILING_INFO_NOT_AVAILABLE:
                    if self.status != self.COMPLETE:
                        raise OpenCLException(err_code, msg='Event must have (status == Event.COMPLETE) before it can be profiled')
                    else:
                        raise OpenCLException(err_code, msg='Queue must be created with profile=True argument in constructor.')
                
                                
                raise OpenCLException(err_code)
            
            return status

    property profile_submitted:
        def __get__(self):
            cdef cl_int err_code
            cdef cl_ulong status

            err_code = clGetEventProfilingInfo(self.event_id, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), & status, NULL)

            if err_code != CL_SUCCESS:
                if err_code == CL_PROFILING_INFO_NOT_AVAILABLE:
                    if self.status != self.COMPLETE:
                        raise OpenCLException(err_code, msg='Event must have (status == Event.COMPLETE) before it can be profiled')
                    else:
                        raise OpenCLException(err_code, msg='Queue must be created with profile=True argument in constructor.')
                
                                
                raise OpenCLException(err_code)
            
            return status
            
        
    def add_callback(self, callback):
        '''
        event.add_callback(callback)
        Registers a user callback function for on completion of the event.
        
        :param callback: must be of the signature callback(event, status)
        '''
        cdef cl_int err_code

        Py_INCREF(callback)
        err_code = clSetEventCallback(self.event_id, CL_COMPLETE, < void *> & pfn_event_notify, < void *> callback) 
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
        
cdef class UserEvent(Event):
    '''
    Creates a user event object.  User events allow applications to enqueue commands that wait on a 
    user event to finish before the command is executed by the device.  
    '''
    def __cinit__(self, context):
        
        cdef cl_int err_code

        cdef cl_context ctx = CyContext_GetID(context)
        self.event_id = clCreateUserEvent(ctx, & err_code)

        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)
        
    def complete(self):
        '''
        Set this event status to complete.
        '''
        cdef cl_int err_code
        
        err_code = clSetUserEventStatus(self.event_id, CL_COMPLETE)
        
        if err_code != CL_SUCCESS:
            raise OpenCLException(err_code)


#===============================================================================
# 
#===============================================================================
cdef api object cl_eventAs_PyEvent(cl_event event_id):
    cdef Event event = < Event > Event.__new__(Event)
    clRetainEvent(event_id)
    event.event_id = event_id
    return event

cdef api cl_event cl_eventFrom_PyEvent(object event):
    return (< Event > event).event_id

cdef api object PyEvent_New(cl_event event_id):
    cdef Event event = < Event > Event.__new__(Event)
    event.event_id = event_id
    return event

cdef api int PyEvent_Check(object event):
    return isinstance(event, Event)
## ############# #### #### #### #### #### #### #### #### #### #### ####
