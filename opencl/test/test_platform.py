'''
Created on Sep 27, 2011

@author: sean
'''
from __future__ import print_function
from opencl import get_platforms, Context, Queue, Program, DeviceMemoryView, empty
from opencl import ContextProperties, global_memory, UserEvent, Event
from opencl.kernel import parse_args
import opencl as cl

import unittest
import ctypes
import numpy as np
from threading import Event as PyEvent
import sys
import os

ctx = None
DEVICE_TYPE = None

def setUpModule():
    global ctx, DEVICE_TYPE
    
    DEVICE_TYPE_ATTR = os.environ.get('DEVICE_TYPE', 'DEFAULT')
    DEVICE_TYPE = getattr(cl.Device, DEVICE_TYPE_ATTR)
    
    ctx = cl.Context(device_type=DEVICE_TYPE)
    print(ctx.devices)


source = """

__kernel void generate_sin(__global float2* a, float scale)
{
    int id = get_global_id(0);
    int n = get_global_size(0);
    float r = (float)id / (float)n;
    
    a[id].x = id;
    a[id].y = native_sin(r) * scale;
}
"""

class Test(unittest.TestCase):
    
    
    def test_platform_constructor(self):
        
        with self.assertRaises(Exception):
            cl.Platform()
            
    def test_device_constructor(self):
        
        with self.assertRaises(Exception):
            cl.Device()
            

    def test_get_platforms(self):
        platforms = get_platforms()

    def test_get_devices(self):
        plat = get_platforms()[0]
        devices = plat.devices
        native_kernels = [dev.has_native_kernel for dev in devices]

    def test_enqueue_native_kernel_refcount(self):
        if not ctx.devices[0].has_native_kernel:
            self.skipTest("Device does not support native kernels")
            
        queue = Queue(ctx, ctx.devices[0])

        def incfoo():
            pass
        
        self.assertEqual(sys.getrefcount(incfoo), 2)
            
        e = cl.UserEvent(ctx)
        queue.enqueue_wait_for_events(e)
        queue.enqueue_native_kernel(incfoo)
        
        self.assertEqual(sys.getrefcount(incfoo), 3)
        
        e.complete()
        
        queue.finish()
        
        self.assertEqual(sys.getrefcount(incfoo), 2)
        
    def test_enqueue_native_kernel(self):
        
        if not ctx.devices[0].has_native_kernel:
            self.skipTest("Device does not support native kernels")
            
        queue = Queue(ctx, ctx.devices[0])

        global foo
        
        foo = 0
        
        def incfoo(arg, op=lambda a, b: 0):
            global foo
            foo = op(foo, arg)
            
        queue.enqueue_native_kernel(incfoo, 4, op=lambda a, b: a + b)
        queue.enqueue_native_kernel(incfoo, 3, op=lambda a, b: a * b)
        
        queue.finish()
        
        self.assertEqual(foo, 12)
        
#        
#    def test_native_kernel_maps_args(self):
#        
#        if not ctx.devices[0].has_native_kernel:
#            self.skipTest("Device does not support native kernels")
#            
#        queue = Queue(ctx, ctx.devices[0])
#        a = cl.empty(ctx, [10], 'f')
#        
#
#        global foo
#        
#        foo = 0
#        
#        def incfoo(arg):
#            global foo
#            
#            print 'arg', arg
#        
#        print "queue.enqueue_native_kernel"
#        queue.enqueue_native_kernel(incfoo, a)
#        
#        print "queue.finish"
#        queue.finish()
#        
#        print "self.assertEqual"
#        self.assertEqual(foo, 12)

class TestDevice(unittest.TestCase):
    
    def _test_device_properties(self):
        
        device = ctx.devices[0]
        print("device_type", device.device_type)
        print("name", device.name)
        print("has_image_support", device.has_image_support)
        print("has_native_kernel", device.has_native_kernel)
        print("max_compute_units", device.max_compute_units)
        print("max_work_item_dimension", device.max_work_item_dimensions)
        print("max_work_item_sizes", device.max_work_item_sizes)
        print("max_work_group_size", device.max_work_group_size)
        print("max_clock_frequency", device.max_clock_frequency, 'MHz')
        print("address_bits", device.address_bits, 'bits')
        print("max_read_image_args", device.max_read_image_args)
        print("max_write_image_args", device.max_write_image_args)
        print("max_image2d_shape", device.max_image2d_shape)
        print("max_image3d_shape", device.max_image3d_shape)
        print("max_parameter_size", device.max_parameter_size, 'bytes')
        print("max_const_buffer_size", device.max_const_buffer_size, 'bytes')
        print("has_local_mem", device.has_local_mem)
        print("local_mem_size", device.local_mem_size, 'bytes')
        print("host_unified_memory", device.host_unified_memory)
        print("available", device.available)
        print("compiler_available", device.compiler_available)
        print("driver_version", device.driver_version)
        print("device_profile", device.profile)
        print("version", device.version)
        print("extensions", device.extensions)


class TestContext(unittest.TestCase):
    def test_properties(self):
        platform = get_platforms()[0]
        
        properties = ContextProperties()
        
        properties.platform = platform
                
        self.assertEqual(platform.name, properties.platform.name)
        
        ctx = Context(device_type=DEVICE_TYPE, properties=properties)

class TestProgram(unittest.TestCase):
        
    def test_program(self):
        
        program = Program(ctx, source=source)
        
        program.build()

    def test_source(self):
        
        program = Program(ctx, source=source)
        
        self.assertEqual(program.source, source)

    def test_binaries(self):
        
        program = Program(ctx, source=source)
        
        self.assertEqual(program.binaries, dict.fromkeys(ctx.devices))
        
        program.build()
        
        binaries = program.binaries
        self.assertIsNotNone(binaries[ctx.devices[0]])
        self.assertEqual(len(binaries[ctx.devices[0]]), program.binary_sizes[0])
        
        program2 = Program(ctx, binaries=binaries)
        
        self.assertIsNone(program2.source)
        
        self.assertEqual(program2.binaries, binaries)
        
    def test_constructor(self):
        
        with self.assertRaises(TypeError):
            Program(None, binaries=None)

        with self.assertRaises(TypeError):
            Program(ctx, binaries={None:None})
        
    def test_devices(self):
        
        program = Program(ctx, source=source)
        
        program.build()
        
class TestKernel(unittest.TestCase):
    
    def test_name(self):
        program = Program(ctx, source=source)
        
        program.build()
        
        generate_sin = program.kernel('generate_sin')
        
        self.assertEqual(generate_sin.name, 'generate_sin')
        
    def test_argtypes(self):

        program = Program(ctx, source=source)
        
        program.build()
        
        generate_sin = program.kernel('generate_sin')
        
        generate_sin.argtypes = [DeviceMemoryView, ctypes.c_float]
        
        with self.assertRaises(TypeError):
            generate_sin.argtypes = [DeviceMemoryView, ctypes.c_float, ctypes.c_float]

    def test_set_args(self):

        program = Program(ctx, source=source)
        
        program.build()
        
        generate_sin = program.kernel('generate_sin')
        
        generate_sin.argtypes = [global_memory(), ctypes.c_float]
        
        buf = empty(ctx, [10], ctype=cl.cl_float2)
        
        queue = Queue(ctx, ctx.devices[0])
        
        generate_sin.set_args(buf, 1.0)
        queue.enqueue_nd_range_kernel(generate_sin, 1, global_work_size=[buf.size])
        
        expected = np.zeros([10], dtype=[('x', np.float32), ('y', np.float32)])
        expected['x'] = np.arange(10)
        expected['y'] = np.sin(expected['x'] / 10)
        
        with buf.map(queue) as host:
            self.assertTrue(np.all(expected['x'] == np.asarray(host)[:, 0]))
            self.assertTrue(np.allclose(expected['y'], np.asarray(host)[:, 1]))

        generate_sin.argnames = ['a', 'scale']
        generate_sin.set_args(a=buf, scale=1.0)
        queue.enqueue_nd_range_kernel(generate_sin, 1, global_work_size=[buf.size])
        
        with buf.map(queue) as host:
            self.assertTrue(np.all(expected['x'] == np.asarray(host)[:, 0]))
            self.assertTrue(np.allclose(expected['y'], np.asarray(host)[:, 1]))
            
        with self.assertRaises(TypeError):
            generate_sin.set_args(a=buf)
            
        generate_sin.__defaults__ = [1.0]
        generate_sin.set_args(a=buf)
        
        queue.enqueue_nd_range_kernel(generate_sin, 1, global_work_size=[buf.size])
        
        with buf.map(queue) as host:
            self.assertTrue(np.all(expected['x'] == np.asarray(host)[:, 0]))
            self.assertTrue(np.allclose(expected['y'], np.asarray(host)[:, 1]))

    def test_call(self):

        expected = np.zeros([10], dtype=[('x', np.float32), ('y', np.float32)])
        expected['x'] = np.arange(10)
        expected['y'] = np.sin(expected['x'] / 10)
        
        program = Program(ctx, source=source)
        
        program.build()
        
        generate_sin = program.kernel('generate_sin')
        
        generate_sin.argtypes = [global_memory(), ctypes.c_float]
        
        buf = empty(ctx, [10], ctype=cl.cl_float2)
        
        queue = Queue(ctx, ctx.devices[0])
        
        size = [buf.size]
        with self.assertRaises(TypeError):
            generate_sin(queue, buf, 1.0)
        
        generate_sin(queue, buf, 1.0, global_work_size=size)
        
        with buf.map(queue) as host:
            self.assertTrue(np.all(expected['x'] == np.asarray(host)[:, 0]))
            self.assertTrue(np.allclose(expected['y'], np.asarray(host)[:, 1]))

        generate_sin.global_work_size = lambda a, scale: [a.size]
        
        generate_sin(queue, buf, 1.0)
        
        with buf.map(queue) as host:
            self.assertTrue(np.all(expected['x'] == np.asarray(host)[:, 0]))
            self.assertTrue(np.allclose(expected['y'], np.asarray(host)[:, 1]))

    def test_parse_args(self):
        
        arglist = parse_args('test', (1, 2, 3), dict(d=4, e=5), ('a', 'b', 'c', 'd', 'e'), ())
        self.assertEqual(arglist, (1, 2, 3, 4, 5))

        arglist = parse_args('test', (1, 2, 3), dict(), ('a', 'b', 'c', 'd', 'e'), (4, 5))
        self.assertEqual(arglist, (1, 2, 3, 4, 5))

        arglist = parse_args('test', (1, 2), dict(c=3), ('a', 'b', 'c', 'd', 'e'), (4, 5))
        self.assertEqual(arglist, (1, 2, 3, 4, 5))

        arglist = parse_args('test', (1, 2), dict(c=3, d=5), ('a', 'b', 'c', 'd', 'e'), (4, 5))
        self.assertEqual(arglist, (1, 2, 3, 5, 5))

        arglist = parse_args('test', (1, 2), dict(c=6, d=6), ('a', 'b', 'c', 'd', 'e'), (4, 5))
        self.assertEqual(arglist, (1, 2, 6, 6, 5))

        arglist = parse_args('test', (), dict(), ('a', 'b', 'c', 'd', 'e'), (1, 2, 3, 4, 5))
        self.assertEqual(arglist, (1, 2, 3, 4, 5))

        
        arglist = parse_args('test', (1, 2, 3, 4, 5), dict(), ('a', 'b', 'c', 'd', 'e'), ())
        self.assertEqual(arglist, (1, 2, 3, 4, 5))
        
        arglist = parse_args('test', (), dict(a=1, b=2, c=3, d=4, e=5), ('a', 'b', 'c', 'd', 'e'), ())
        self.assertEqual(arglist, (1, 2, 3, 4, 5))
        
        with self.assertRaises(TypeError):
            arglist = parse_args('test', (), dict(), ('a', 'b'), ())

        with self.assertRaises(TypeError):
            arglist = parse_args('test', (1), dict(a=1), ('a', 'b'), ())

        with self.assertRaises(TypeError):
            arglist = parse_args('test', (), dict(b=1), ('a', 'b'), (2))
            
        with self.assertRaises(TypeError):
            arglist = parse_args('test', (1, 2, 3), dict(), ('a', 'b'), ())

class TestEvent(unittest.TestCase):
    
    def test_status(self):
        
        event = UserEvent(ctx)
        
        self.assertEqual(event.status, Event.SUBMITTED)
        event.complete()
        self.assertEqual(event.status, Event.COMPLETE)
        
    def test_wait(self):
        
        event = UserEvent(ctx)

        queue = Queue(ctx, ctx.devices[0])
        
        queue.enqueue_wait_for_events(event)
        
        event2 = queue.marker()
        
        self.assertEqual(event.status, Event.SUBMITTED)
        self.assertEqual(event2.status, Event.QUEUED)
        
        event.complete()
        self.assertEqual(event.status, Event.COMPLETE)
        
        event2.wait()
        self.assertEqual(event2.status, Event.COMPLETE)

    def test_callback(self):
        self.callback_called = False
        self.py_event = PyEvent()
        
        def callback(event, status):
            self.callback_called = True
            self.py_event.set()
        
        event = UserEvent(ctx)

        queue = Queue(ctx, ctx.devices[0])
        
        queue.enqueue_wait_for_events(event)
        
        event2 = queue.marker()
        event2.add_callback(callback)
        
        self.assertEqual(event.status, Event.SUBMITTED)
        self.assertEqual(event2.status, Event.QUEUED)
        
        self.assertFalse(self.callback_called)
        event.complete()
        self.assertEqual(event.status, Event.COMPLETE)
        
        event2.wait()
        self.assertEqual(event2.status, Event.COMPLETE)
        
        event_is_set = self.py_event.wait(2)
        
        self.assertTrue(event_is_set, 'timed out waiting for callback')

        self.assertTrue(self.callback_called)

        
if __name__ == '__main__':
    
    unittest.main()
