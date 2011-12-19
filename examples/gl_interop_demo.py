'''
Created on Dec 8, 2011

@author: sean
'''

from OpenGL.GL import *
from OpenGL.GLUT import *
import opencl as cl
import numpy as np


generate_sin_source = '''
#line 14 "gl_interop_demo.py"
__kernel void generate_sin(__global float2* a) {
    uint gid = get_global_id(0);
    uint n = get_global_size(0);
    float r = (float)gid / (float)n;
    
    float x = r * 16.0f * 3.1415f;
    
    a[gid].x = r * 2.0f - 1.0f;
    a[gid].y = native_sin(x);
}
'''

n_vertices = 100
coords_dev = None

generate_sin = None
 
def initialize():
    global generate_sin, coords_dev, n_vertices
    
    ctx = cl.gl.context()

    if generate_sin is None:
        program = cl.Program(ctx, generate_sin_source).build()
        generate_sin = program.generate_sin
        
        generate_sin.argnames = 'a',
        generate_sin.argtypes = cl.global_memory(cl.cl_float2),
        generate_sin.global_work_size = lambda a: a.shape
    
    coords_dev = cl.gl.empty_gl(ctx, [n_vertices], ctype=cl.cl_float2)
    
    glClearColor(1, 1, 1, 1)
    glColor(0, 0, 1)
    
    queue = cl.Queue(ctx)
    
    with cl.gl.acquire(queue, coords_dev):
        generate_sin(queue, coords_dev)
        
    glEnableClientState(GL_VERTEX_ARRAY)
    
def display():
    global coords_dev, n_vertices

    glClear(GL_COLOR_BUFFER_BIT)
    
    vbo = cl.gl.get_gl_name(coords_dev)
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(2, GL_FLOAT, 0, None)
    glDrawArrays(GL_LINE_STRIP, 0, n_vertices)
    
    glFlush()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)

def main():
    import sys
    glutInit(sys.argv)
    if len(sys.argv) > 1:
        n_vertices = int(sys.argv[1])
    glutInitWindowSize(800, 160)
    glutInitWindowPosition(0, 0)
    glutCreateWindow('OpenCL/OpenGL Interop Tutorial: Sin Generator')
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    initialize()
    glutMainLoop()
    
if __name__ == '__main__':
    main()
