
from _cl cimport *
## API FUNCTIONS #### #### #### #### #### #### #### #### #### #### ####
## ############# #### #### #### #### #### #### #### #### #### #### ####
#===============================================================================
# 
#===============================================================================

cdef api cl_platform_id CyPlatform_GetID(object py_platform)

cdef api object CyPlatform_Create(cl_platform_id platform_id)

#===============================================================================
# 
#===============================================================================

cdef api cl_device_id CyDevice_GetID(object py_device)
cdef api int CyDevice_Check(object py_device)

cdef api object CyDevice_Create(cl_device_id device_id)

