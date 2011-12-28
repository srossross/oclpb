
cimport _cl as cl 

import ctypes
import _ctypes
#
#    ctypedef int8_t cl_char
cl_char = ctypes.c_char
#
#    ctypedef uint8_t cl_uchar
cl_uchar = ctypes.c_ubyte
#
int_types = [ctypes.c_int8, ctypes.c_int16, ctypes.c_int32, ctypes.c_int64]
uint_types = [ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_uint64]

for int_type in int_types:
    if ctypes.sizeof(int_type) == sizeof(cl.cl_short):
        cl_short = int_type

for int_type in uint_types:
    if ctypes.sizeof(int_type) == sizeof(cl.cl_ushort):
        cl_ushort = int_type

for int_type in int_types:
    if ctypes.sizeof(int_type) == sizeof(cl.cl_int):
        cl_int = int_type

for int_type in uint_types:
    if ctypes.sizeof(int_type) == sizeof(cl.cl_uint):
        cl_uint = int_type

for int_type in int_types:
    if ctypes.sizeof(int_type) == sizeof(cl.cl_long):
        cl_long = int_type

for int_type in uint_types:
    if ctypes.sizeof(int_type) == sizeof(cl.cl_ulong):
        cl_ulong = int_type

for int_type in uint_types:
    if ctypes.sizeof(int_type) == sizeof(cl.cl_half):
        cl_half = int_type

#
#    ctypedef float cl_float
cl_float = ctypes.c_float
#
#    ctypedef double cl_double
cl_double = ctypes.c_double
#
#    ctypedef int8_t cl_char2[2]

array_type = type(_ctypes.Array)


class typed_property(object):
    pass

_MP = {'x':1, 'y':2, 'z':3, 'w':4}

class _cl_vector_methods(object):

    def __getattr__(self, attr):
        _type_ = getattr(type(self), attr)
        if attr[0] in 'xyzw':
            elems = [self[_MP[c]] for c in attr[:]]
        else:
            elems = [self[int(c, 16)] for c in attr[1:]]
        if len(elems) == 1:
            return _type_(elems[0])
        
        return _type_(*elems)


class cl_vector(array_type):
    def __new__(cls, name, type, length):
        return array_type.__new__(cls, name, (_cl_vector_methods, _ctypes.Array), {'_type_':type, '_length_':length})
    
    def __getattr__(self, attr):
        if attr[0] in 'xyzw':
            if len(attr) == 1:
                return self._type_
            else:
                if set(attr) - set('xyzw'[:self._length_]):
                    raise AttributeError('type %r has no attribute %s' % (self, attr))
                return cl_vector('cl_char%i' % len(attr), self._type_, len(attr))
        elif attr[0] == 's':
            if set(attr[1:]) - set('0123456789abcdef'[:self._length_]):
                raise AttributeError('type %r has no attribute %s' % (self, attr))
            if len(attr) == 2:
                return self._type_
            else:
                return cl_vector('cl_char%i' % (len(attr) - 1), self._type_, len(attr) - 1)
    
    def __getitem__(self, index_type):
        return self._type_


try:
    import copy_reg as copyreg 
except: #Python3
    import copyreg 

def pickle_cl_vector(vector_obj):
    constructor = cl_vector
    args = (vector_obj.__name__, vector_obj._type_, vector_obj._length_)
    state = None
    
    return (constructor, args, state, None, None)

    
copyreg.pickle(cl_vector, pickle_cl_vector)
            
    

cl_char2 = cl_vector('cl_char2', cl_char, 2)
cl_char4 = cl_vector('cl_char4', cl_char, 4)
cl_char8 = cl_vector('cl_char8', cl_char, 8)
cl_char16 = cl_vector('cl_char16', cl_char, 16)

cl_uchar2 = cl_vector('cl_uchar2', cl_uchar, 2)
cl_uchar4 = cl_vector('cl_uchar4', cl_uchar, 4)
cl_uchar8 = cl_vector('cl_uchar8', cl_uchar, 8)
cl_uchar16 = cl_vector('cl_uchar16', cl_uchar, 16)

cl_short2 = cl_vector('cl_short2', cl_short, 2)
cl_short4 = cl_vector('cl_short4', cl_short, 4)
cl_short8 = cl_vector('cl_short8', cl_short, 8)
cl_short16 = cl_vector('cl_short16', cl_short, 16)

cl_ushort2 = cl_vector('cl_ushort2', cl_ushort, 2)
cl_ushort4 = cl_vector('cl_ushort4', cl_ushort, 4)
cl_ushort8 = cl_vector('cl_ushort8', cl_ushort, 8)
cl_ushort16 = cl_vector('cl_ushort16', cl_ushort, 16)

cl_int2 = cl_vector('cl_int2', cl_int, 2)
cl_int4 = cl_vector('cl_int4', cl_int, 4)
cl_int8 = cl_vector('cl_int8', cl_int, 8)
cl_int16 = cl_vector('cl_int16', cl_int, 16)

cl_uint2 = cl_vector('cl_uint2', cl_uint, 2)
cl_uint4 = cl_vector('cl_uint4', cl_uint, 4)
cl_uint8 = cl_vector('cl_uint8', cl_uint, 8)
cl_uint16 = cl_vector('cl_uint16', cl_uint, 16)

cl_long2 = cl_vector('cl_long2', cl_long, 2)
cl_long4 = cl_vector('cl_long4', cl_long, 4)
cl_long8 = cl_vector('cl_long8', cl_long, 8)
cl_long16 = cl_vector('cl_long16', cl_long, 16)

cl_ulong2 = cl_vector('cl_ulong2', cl_ulong, 2)
cl_ulong4 = cl_vector('cl_ulong4', cl_ulong, 4)
cl_ulong8 = cl_vector('cl_ulong8', cl_ulong, 8)
cl_ulong16 = cl_vector('cl_ulong16', cl_ulong, 16)

cl_float2 = cl_vector('cl_float2', cl_float, 2)
cl_float4 = cl_vector('cl_float4', cl_float, 4)
cl_float8 = cl_vector('cl_float8', cl_float, 8)
cl_float16 = cl_vector('cl_float16', cl_float, 16)

cl_double2 = cl_vector('cl_double2', cl_double, 2)
cl_double4 = cl_vector('cl_double4', cl_double, 4)
cl_double8 = cl_vector('cl_double8', cl_double, 8)
cl_double16 = cl_vector('cl_double16', cl_double, 16)
