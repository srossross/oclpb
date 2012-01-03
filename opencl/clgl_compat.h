#if defined __APPLE__ || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenCL/cl_gl_ext.h>
#include <OpenCL/cl_gl.h>


#else

#define CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE 0

static void * CGLGetCurrentContext(void){return NULL;}
static void * CGLGetShareGroup(void *f){return NULL;}

//#include <GL/gl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>

#endif
