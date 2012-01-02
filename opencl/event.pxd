
from _cl cimport *
#===============================================================================
# 
#===============================================================================
cdef api object cl_eventAs_PyEvent(cl_event event_id)
cdef api cl_event cl_eventFrom_PyEvent(object event)
cdef api object PyEvent_New(cl_event event_id)
cdef api int PyEvent_Check(object event)

## ############# #### #### #### #### #### #### #### #### #### #### ####
