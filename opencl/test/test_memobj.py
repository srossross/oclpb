'''
Created on Dec 24, 2011

@author: sean
'''
from __future__ import print_function
from opencl import Platform, get_platforms, Context, Device, Queue, Program, DeviceMemoryView, empty
from opencl import ContextProperties, global_memory, UserEvent, Event
from opencl.kernel import parse_args
import opencl as cl

import unittest
import ctypes
from ctypes import c_int, c_float, sizeof
import numpy as np
import gc
from threading import Event as PyEvent
import sys
import os

ctx = None

def setUpModule():
    global ctx
    
    
    DEVICE_TYPE_ATTR = os.environ.get('DEVICE_TYPE', 'DEFAULT')
    DEVICE_TYPE = getattr(cl.Device, DEVICE_TYPE_ATTR)
    
    ctx = cl.Context(device_type=DEVICE_TYPE)
    print(ctx.devices)
        
class TestBuffer(unittest.TestCase):
    
    def test_size(self):     
        buf = empty(ctx, [4])
        
        self.assertEqual(buf._refcount, 1)
        
        self.assertEqual(len(buf), 4 / buf.itemsize)
        self.assertEqual(buf.mem_size, 4)
        
        layout = buf.array_info
        
        self.assertEqual(layout[:4], [4, 0, 0, 4]) #shape
        self.assertEqual(layout[4:], [1, 0, 0, 0]) #strides
        
    def test_local_memory(self):
        a = np.array([[1, 2], [3, 4]])
        view_a = memoryview(a)
        clmem = DeviceMemoryView.from_host(ctx, a)
        
        self.assertEqual(clmem.format, view_a.format)
        self.assertEqual(clmem.shape, view_a.shape)
        self.assertEqual(clmem.strides, view_a.strides)
        
    def test_from_host(self):
        a = np.array([[1, 2], [3, 4]])
        view_a = memoryview(a)
        clmem = DeviceMemoryView.from_host(ctx, a)
        
        self.assertEqual(clmem.format, view_a.format)
        self.assertEqual(clmem.shape, view_a.shape)
        self.assertEqual(clmem.strides, view_a.strides)

    def test_from_host_no_copy(self):
        
        a = np.array([[1, 2], [3, 4]])
        
        refcount = sys.getrefcount(a)
        
        clmem = cl.from_host(ctx, a, copy=False)
        
#        event = PyEvent()
#        def set_event(mem):
#            event.set()
            
#        clmem.add_destructor_callback(set_event)
        
        self.assertEqual(refcount + 1, sys.getrefcount(a))
        
        del clmem
        gc.collect()
        
#        self.assertTrue(event.wait(1), 'event timed out. destructor_callback not called')
        
        self.assertEqual(refcount, sys.getrefcount(a))
        
        clmem = cl.from_host(ctx, a, copy=False)
        
        view_a = memoryview(a)
        
        self.assertEqual(clmem.format, view_a.format)
        self.assertEqual(clmem.shape, view_a.shape)
        self.assertEqual(clmem.strides, view_a.strides)
        
        queue = cl.Queue(ctx)
        
        if queue.device.host_unified_memory:
            a[0, 0] = 100
            with clmem.map(queue) as view:
                b = np.asarray(view) 
                self.assertEqual(b[0, 0], 100)
        else:
            #TODO: should there be a test here?
            pass
        
    def test_read_write(self):
        a = np.array([[1, 2], [3, 4]])
        clbuf = DeviceMemoryView.from_host(ctx, a)
        
        queue = Queue(ctx, ctx.devices[0])
        
        out = np.zeros_like(a)
        
        clbuf.read(queue, out, blocking=True)
        
        self.assertTrue(np.all(out == a))
        
        clbuf.write(queue, a + 1, blocking=True)

        clbuf.read(queue, out, blocking=True)
        
        self.assertTrue(np.all(out == a + 1))
        
    def test_map(self):
        
        a = np.array([[1, 2], [3, 4]])
        view_a = memoryview(a)
        
        clbuf = DeviceMemoryView.from_host(ctx, a)
    
        queue = Queue(ctx, ctx.devices[0])
        
        self.assertEqual(clbuf._mapcount, 0)
        
        with clbuf.map(queue, writeable=False) as buf:
            self.assertEqual(clbuf._mapcount, 1)
        
            self.assertEqual(buf.format, view_a.format)
            self.assertEqual(buf.shape, view_a.shape)
            self.assertEqual(buf.strides, view_a.strides)
            
            b = np.asarray(buf)
            self.assertTrue(np.all(b == a))
        
            self.assertTrue(buf.readonly)
        
#        self.assertEqual(clbuf._mapcount, 0)
        
        with clbuf.map(queue, readable=False) as buf:
            self.assertEqual(clbuf._mapcount, 1)
            b = np.asarray(buf)
            b[::] = a[::-1]
            
            self.assertFalse(buf.readonly)
            
#        self.assertEqual(clbuf._mapcount, 0)
        
        with clbuf.map(queue, writeable=False) as buf:
            self.assertEqual(clbuf._mapcount, 1)
            b = np.asarray(buf)
            self.assertTrue(np.all(b == a[::-1]))
            
    def test_refcount(self):

        a = np.array([[1, 2], [3, 4]])
        
        clbuf = DeviceMemoryView.from_host(ctx, a)

        self.assertEqual(clbuf._refcount, 1)
        
        new_buf = clbuf[:, :-1]
        
        self.assertEqual(clbuf._refcount, 2)
        
        del new_buf
        gc.collect()
        
        self.assertEqual(clbuf._refcount, 1)
        
        self.assertEqual(clbuf.base, None)
        
        #create sub_buffer
        new_buf = clbuf[1, :]

        self.assertEqual(clbuf._refcount, 2)
        
        del new_buf
        gc.collect()
        
        self.assertEqual(clbuf._refcount, 1)

        queue = Queue(ctx)
        with clbuf.map(queue) as host:
            self.assertEqual(clbuf._refcount, 1)
            
        self.assertEqual(clbuf._refcount, 2, "unmap increments the refcount")
        
        del host
        gc.collect()
        
        #GPU may not decrement the ref count
        #unless finish is called
        queue.finish()
        self.assertEqual(clbuf._refcount, 1)
            
        event = PyEvent()
        def callback(mem):
            event.set()
        
        
        #clbuf.add_destructor_callback(callback)
        
        #del clbuf
        #gc.collect()
        
        #timed_out = not event.wait(1)
        #self.assertFalse(timed_out)

            
    def test_get_slice(self):
        
        queue = Queue(ctx, ctx.devices[0])   
        a = np.array([1, 2, 3, 4])
        
        clbuf = DeviceMemoryView.from_host(ctx, a)

        self.assertEqual(clbuf._refcount, 1)
        
        new_buf = clbuf[::2]
        
        with new_buf.map(queue) as buf:
            b = np.asanyarray(buf)
            self.assertTrue(np.all(b == a[::2]))
            
        new_buf = clbuf[1::2]
        
        with new_buf.map(queue) as buf:
            b = np.asanyarray(buf)
            self.assertTrue(np.all(b == a[1::2]))

        new_buf = clbuf[::-1]
        
        with new_buf.map(queue) as buf:
            b = np.asanyarray(buf)
            self.assertTrue(np.all(b == a[::-1]))
            
        
    def test_dim_reduce(self):
        queue = Queue(ctx, ctx.devices[0])   
        a = np.array([[1, 2], [3, 4], [5, 6]])
    
        view = DeviceMemoryView.from_host(ctx, a)
        
        new_view = view[:, 0]
        
        self.assertEqual(new_view.ndim, a[:, 0].ndim)
        self.assertEqual(new_view.shape, a[:, 0].shape)
        self.assertEqual(new_view.offset_, 0)
        self.assertEqual(new_view.strides, a[:, 0].strides)
        
        with new_view.map(queue) as buf:
            b = np.asarray(buf)
            self.assertTrue(np.all(b == a[:, 0]))

        new_view = view[:, 1]
        
        with new_view.map(queue) as buf:
            b = np.asarray(buf)
            self.assertTrue(np.all(b == a[:, 1]))

        new_view = view[0, :]
        
        with new_view.map(queue) as buf:
            b = np.asarray(buf)
            self.assertTrue(np.all(b == a[0, :]))

        new_view = view[1, :]
        
        with new_view.map(queue) as buf:
            b = np.asarray(buf)
            self.assertTrue(np.all(b == a[1, :]))

    def test_getitem(self):
        queue = Queue(ctx, ctx.devices[0])   
        a = np.array([[1, 2], [3, 4]])
        
        clbuf = DeviceMemoryView.from_host(ctx, a)

        with self.assertRaises(IndexError):
            clbuf[1, 1, 1]
        
        self.assertEqual(clbuf._refcount, 1)
        
        new_buf = clbuf[:, :-1]
        
        self.assertEqual(clbuf._refcount, 2)
        
        mapp = new_buf.map(queue)
        
        with mapp as buf:
            
            b = np.asanyarray(buf)
            self.assertTrue(np.all(b == a[:, :-1]))
        
        del buf
        del new_buf
        gc.collect()
        
        new_buf = clbuf[:, 1:]
        
        with new_buf.map(queue) as buf:
            b = np.asanyarray(buf)
            self.assertTrue(np.all(b == a[:, 1:]))

        new_buf = clbuf[1:, :]
        
        with new_buf.map(queue) as buf:
            b = np.asanyarray(buf)
            self.assertTrue(np.all(b == a[1:, :]))
        
    def test_is_contiguous(self):
        
        a = np.array([[1, 2], [3, 4]])
        
        clbuf = DeviceMemoryView.from_host(ctx, a)
        
        self.assertTrue(clbuf.is_contiguous)
        
        self.assertFalse(clbuf[:, 1:].is_contiguous)
        self.assertFalse(clbuf[::2, :].is_contiguous)
        self.assertFalse(clbuf[:, ::2].is_contiguous)
        
    def test_copy_contig(self):

        queue = Queue(ctx, ctx.devices[0])   
        a = np.array([[1, 2], [3, 4]])
        
        clbuf = DeviceMemoryView.from_host(ctx, a)
        
        copy_of = clbuf.copy(queue)
        queue.barrier()
        with copy_of.map(queue) as cpy:
            b = np.asarray(cpy)
            self.assertTrue(np.all(a == b))
            
    def test_copy_1D(self):
        
        queue = Queue(ctx, ctx.devices[0])   
        a = np.array([1, 2, 3, 4])
        
        clbuf = DeviceMemoryView.from_host(ctx, a)
        
        copy_of = clbuf[::2].copy(queue)
        
        with copy_of.map(queue) as cpy:
            b = np.asarray(cpy)
            self.assertTrue(np.all(a[::2] == b))
            
        copy_of = clbuf[1::2].copy(queue)
            
        with copy_of.map(queue) as cpy:
            b = np.asarray(cpy)
            
            self.assertTrue(np.all(a[1::2] == b))
            
        copy_of = clbuf[1:-1].copy(queue)
            
        with copy_of.map(queue) as cpy:
            b = np.asarray(cpy)
            self.assertTrue(np.all(a[1:-1] == b))

    def test_copy_2D(self):


        queue = Queue(ctx, ctx.devices[0])   
        a = np.arange(6 * 6).reshape([6, 6])
        
        clbuf = DeviceMemoryView.from_host(ctx, a)
        
        slices = [
                  (slice(None, None, 2), slice(None, None, 2)),
                  (slice(1, None, None), slice(1, None, None)),
                  (slice(None, None, None), slice(1, None, None)),
                  (slice(1, None, None), slice(None, None, None)),
                  
                  (slice(1, None, None), slice(0, None, 2)),
                  (slice(None, None, 2), slice(1, None, 2)),
                  (slice(1, None, 2), slice(None, None, 2)),
                  ]
        
        for idx0, idx1 in slices:
            
            expected = a[idx0, idx1]
            sub_buf = clbuf[idx0, idx1]
            copy_of = sub_buf.copy(queue)
        
            with copy_of.map(queue) as cpy:
                b = np.asarray(cpy)
                expected = a[idx0, idx1]
                self.assertTrue(np.all(expected == b), (idx0, idx1))
                
    @unittest.expectedFailure     
    def test_copy_2D_negative_stride(self):


        queue = Queue(ctx, ctx.devices[0])   
        a = np.arange(4 * 4).reshape([4, 4])
        
        clbuf = DeviceMemoryView.from_host(ctx, a)
        
        slices = [(slice(None, None, -2), slice(None, None, -2)),
                  
                  (slice(1, None, -1), slice(1, None, -1)),
                  (slice(None, None, None), slice(1, None, -1)),
                  (slice(1, None, -1), slice(None, None, -1)),
                  
                  (slice(1, None, -2), slice(1, None, -2)),
                  (slice(None, None, -2), slice(1, None, -2)),
                  (slice(1, None, -2), slice(None, None, -2)),
                  ]
        
        for idx0, idx1 in slices:
            copy_of = clbuf[idx0, idx1].copy(queue)
        
            with copy_of.map(queue) as cpy:
                b = np.asarray(cpy)
                expected = a[idx0, idx1]
                self.assertTrue(np.all(expected == b))

    def test_broadcast_0D(self):
        
        with self.assertRaises(TypeError):
            cl.broadcast(None, [1])
            
        one = cl.from_host(ctx, c_int(1))
        
        a = cl.broadcast(one, [10, 10])
        self.assertEqual(a.shape, (10, 10))
        self.assertEqual(a.strides, (0, 0))
        
        queue = cl.Queue(ctx)
        with a.map(queue) as view:
            b = np.asarray(view)
            self.assertEqual(b.shape, (10, 10))
            self.assertEqual(b.strides, (0, 0))
    
    def test_broadcast_2D(self):
        
        with self.assertRaises(TypeError):
            cl.broadcast(None, [1])
            
        npa = np.arange(10, dtype=c_float)
        z = np.zeros([10, 1])
        
        ten = cl.from_host(ctx, npa)
        
        a = cl.broadcast(ten, [10, 10])
        self.assertEqual(a.shape, (10, 10))
        self.assertEqual(a.strides, (0, sizeof(c_float)))
        
        queue = cl.Queue(ctx)
        with a.map(queue) as view:
            b = np.asarray(view)
            self.assertEqual(b.shape, (10, 10))
            self.assertEqual(b.strides, (0, sizeof(c_float)))
            self.assertTrue(np.all(b == z + npa))
    
    def test_ctype(self):
        
        a = cl.empty(ctx, [2], cl.cl_float2)
        
        b = a[1:]
        
        self.assertIs(a.ctype, b.ctype)
    
            
class TestImage(unittest.TestCase):
    def test_supported_formats(self):
        image_format = cl.ImageFormat.supported_formats(ctx)[0]
        
#        print(cl.ImageFormat.CHANNEL_ORDERS)
        format_copy = cl.ImageFormat.from_ctype(image_format.ctype)
        
        self.assertEqual(image_format, format_copy)
        
    def test_empty(self):

        image_format = cl.ImageFormat('CL_RGBA', 'CL_UNSIGNED_INT8')
        
        image = cl.empty_image(ctx, [4, 4], image_format)
        
        self.assertEqual(image.type, cl.Image.IMAGE2D)
        
        self.assertEqual(image.image_format, image_format)
        self.assertEqual(image.image_width, 4)
        self.assertEqual(image.image_height, 4)
        self.assertEqual(image.image_depth, 1)

    def test_empty_3d(self):

        image_format = cl.ImageFormat('CL_RGBA', 'CL_UNSIGNED_INT8')
        
        image = cl.empty_image(ctx, [4, 4, 4], image_format)
        
        self.assertEqual(image.type, cl.Image.IMAGE3D)
        self.assertEqual(image.image_format, image_format)
        self.assertEqual(image.image_width, 4)
        self.assertEqual(image.image_height, 4)
        self.assertEqual(image.image_depth, 4)

        
    def test_map(self):

        image_format = cl.ImageFormat('CL_RGBA', 'CL_UNSIGNED_INT8')
        
        image = cl.empty_image(ctx, [4, 4], image_format)
        
        queue = Queue(ctx)   

        with image.map(queue) as img:
            self.assertEqual(img.format, 'T{B:r:B:g:B:b:B:a:}')
            self.assertEqual(img.ndim, 2)
            self.assertEqual(img.shape, (4, 4))
            
            array = np.asarray(img)
            array['r'] = 1
            
        with image.map(queue) as img:
            array = np.asarray(img)
            self.assertTrue(np.all(array['r'] == 1))
            
if __name__ == '__main__':
    unittest.main()


