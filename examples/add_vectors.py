'''
Created on Dec 19, 2011

@author: sean
'''

import opencl as cl
import numpy as np

def main():
    
    size = 10
    a = np.random.rand(size).astype('f')
    b = np.random.rand(size).astype('f')
    
    ctx = cl.Context()
    queue = cl.Queue(ctx)
    
    cla = cl.from_host(ctx, a, copy=True)
    clb = cl.from_host(ctx, b, copy=True)
    clc = cl.empty(ctx, [size], ctype='f')
    
    prg = cl.Program(ctx, """
        __kernel void add(__global const float *a,
        __global const float *b, __global float *c)
        {
          int gid = get_global_id(0);
          c[gid] = a[gid] + b[gid];
        }
        """).build()
    
    add = prg.add
    add.argtypes = cl.global_memory('f'), cl.global_memory('f'), cl.global_memory('f')
    add.argnames = 'a', 'b', 'c'
    add.global_work_size = lambda a: a.shape
    
    add(queue, a=cla, b=clb, c=clc)
    
    with clc.map(queue) as view:
        print "view is a python memoryview object", view
        
        arr = np.asarray(view)
        
        print "Answer should be zero:"
        print (arr - (a + b)).sum()
    
    
if __name__ == '__main__':
    main()

