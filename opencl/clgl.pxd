
from _cl cimport * 

cdef extern from "clgl_compat.h":
    enum gl_context_properties:
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE


    void * CGLGetCurrentContext()
    void * CGLGetShareGroup(void *)
    
    ctypedef unsigned cl_GLuint
    ctypedef int cl_GLint
    ctypedef void cl_GLvoid
    #ctypedef size_t size_t
    #ctypedef size_t size_t
    
    enum cl_gl_error:
        CL_INVALID_GL_OBJECT
        
    enum cl_gl_object_type:
        CL_GL_OBJECT_BUFFER
        CL_GL_OBJECT_TEXTURE2D
        CL_GL_OBJECT_TEXTURE3D
        CL_GL_OBJECT_RENDERBUFFER
        
    ctypedef int cl_GLenum
    
    cdef cl_GLenum GL_ARRAY_BUFFER
    cdef cl_GLenum GL_STATIC_DRAW
    cdef cl_GLenum GL_NO_ERROR
    cdef cl_GLenum GL_TEXTURE_2D
    cdef cl_GLenum GL_TEXTURE_3D
    cdef cl_GLenum GL_UNSIGNED_BYTE
    
    
    #void glGenBuffers(size_t, cl_GLuint *)
    #void glBindBuffer(cl_GLenum, cl_GLuint)
    #void glBufferData(cl_GLenum, size_t, cl_GLvoid * , cl_GLenum)
    
    #cl_GLenum glGetError()
     
    # void glEnable(cl_GLenum) 
    # void glGenTextures(size_t, cl_GLuint *)
    # void glBindTexture(cl_GLenum, cl_GLuint)

    # void glTexImage2D(cl_GLenum, cl_GLint, cl_GLint, size_t, size_t, cl_GLint, cl_GLenum, cl_GLenum, cl_GLvoid *)
    # void glTexImage3D(cl_GLenum, cl_GLint, cl_GLint, size_t, size_t, size_t, cl_GLint, cl_GLenum, cl_GLenum, cl_GLvoid *)

    cdef cl_GLenum GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, CL_RGBA, GL_BGRA, \
                GL_UNSIGNED_INT_8_8_8_8_REV, CL_BGRA, GL_RGBA16, GL_RGBA8I, \
                GL_RGBA8I_EXT, GL_RGBA16I, GL_RGBA16I_EXT, GL_RGBA32I, GL_RGBA32I_EXT, \
                GL_RGBA8UI, GL_RGBA8UI_EXT, GL_RGBA16UI, GL_RGBA16UI_EXT, GL_RGBA32UI, \
                GL_RGBA32UI_EXT, GL_RGBA16F, GL_RGBA16F_ARB, GL_RGBA32F, GL_RGBA32F_ARB


    #cl_gl.h
    
    cl_mem clCreateFromGLBuffer(cl_context, cl_mem_flags, unsigned, cl_int *)
    
    cl_mem clCreateFromGLTexture2D(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *)
    cl_mem clCreateFromGLTexture3D(cl_context, cl_mem_flags, cl_GLenum, cl_GLint, cl_GLuint, cl_int *)

    cl_int clGetGLObjectInfo(cl_mem memobj, cl_gl_object_type * gl_object_type, cl_GLuint * gl_object_name)
    
    cl_int clEnqueueAcquireGLObjects(cl_command_queue, cl_uint, cl_mem * , cl_uint, cl_event * , cl_event *)
    cl_int clEnqueueReleaseGLObjects(cl_command_queue, cl_uint, cl_mem * , cl_uint, cl_event * , cl_event *)
    

   
