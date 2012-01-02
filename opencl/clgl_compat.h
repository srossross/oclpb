#if defined __APPLE__ || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenCL/cl_gl_ext.h>
#include <OpenCL/cl_gl.h>


#else

#define CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE 0

static void * CGLGetCurrentContext(){reutrn NULL;}
static void * CGLGetShareGroup(void *){reutrn NULL;}

#include <gl/gl.h>
#include <cl/cl_gl.h>
#include <cl/cl_gl_ext.h>

#endif
