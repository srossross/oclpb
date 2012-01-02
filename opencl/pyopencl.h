
#include "cl_mem_api.h"
#include "context_api.h"
#include "copencl_api.h"
#include "event_api.h"
#include "kernel_api.h"
#include "program_api.h"
#include "queue_api.h"

import_opencl(void) {
    if (import_opencl__queue() < 0) goto bad;
    if (import_opencl__cl_mem() < 0) goto bad;
    if (import_opencl__context() < 0) goto bad;
    if (import_opencl__copencl() < 0) goto bad;
    if (import_opencl__event() < 0) goto bad;
    if (import_opencl__kernel() < 0) goto bad;
    if (import_opencl__program() < 0) goto bad;
    
    return 0;
    
    bad:
        return -1;
}