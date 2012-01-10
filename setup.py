'''
Created on Sep 24, 2011

@author: sean
'''

from setuptools import setup, find_packages, Extension
from os.path import join, isfile, isdir
import os
import sys
from warnings import warn

try:
    from Cython.Distutils.build_ext import build_ext
    cmdclass = {'build_ext': build_ext}
except ImportError:
    cmdclass = { }

DONT_USE_CYTHON = os.environ.get('CL_NO_CYTHON')
if DONT_USE_CYTHON:
    cmdclass = { }

if 'darwin' in sys.platform:
    flags = dict(extra_link_args=['-framework', 'OpenCL'])
elif sys.platform.startswith('win32'):
    
    include_dirs = []
    library_dirs = []
    
    AMDAPPSDKROOT = os.environ.get('AMDAPPSDKROOT', r'does\not\exist')
    if isdir(AMDAPPSDKROOT):
        include_dirs.append(join(AMDAPPSDKROOT, 'include'))
        library_dirs.append(join(AMDAPPSDKROOT, 'lib'))
        
    if isdir(r'C:\Program Files\ATI Stream'):
        include_dirs.append(r'C:\Program Files\ATI Stream\include')
        library_dirs.append(r'C:\Program Files\ATI Stream\lib\x86')
    
        
    flags = dict(libraries=['OpenCL'], include_dirs=include_dirs, library_dirs=library_dirs)
    
else:
    AMDAPPSDKROOT = os.environ.get('AMDAPPSDKROOT', '/usr/local')
    
    flags = dict(libraries=['OpenCL'], include_dirs=[join(AMDAPPSDKROOT, 'include')], library_dirs=[join(AMDAPPSDKROOT, 'lib')])

extension = lambda name, ext: Extension('.'.join(('opencl', name)), [join('opencl', name + ext)], **flags)
pyx_extention_names = [name[:-4] for name in os.listdir('opencl') if name.endswith('.pyx')]

try:
    import OpenGL.GL
    have_opengl = True
except ImportError as err:
    have_opengl = False
    print err
    
if os.environ.get('NO_OPENGL'):
    have_opengl = False
    
if not have_opengl:
    pyx_extention_names.remove('clgl')

if cmdclass:
    ext_modules = [extension(name, '.pyx') for name in pyx_extention_names]
else:
    warn("Cython not installed using pre-cythonized files", UserWarning, stacklevel=1)
    for name in pyx_extention_names:
        required_c_file = join('opencl', name + '.c')
        if not isfile(join('opencl', name + '.c')):
            raise Exception("Cython is required to build a c extension from a PYX file (solution get cython or checkout a release branch)")
    
    ext_modules = [extension(name, '.c') for name in pyx_extention_names]

try:
    long_description = open('README.rst').read()
except IOError as err:
    long_description = str(err)
try:
    version_str = open('version.txt').read()
except IOError as err:
    version_str = '???'

setup(
    name='opencl-for-python',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    version=version_str,
    author='Enthought, Inc.',
    author_email='srossross@enthought.com',
    url='http://srossross.github.com/oclpb',
    classifiers=[c.strip() for c in """\
        Development Status :: 5 - Production/Stable
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: BSD License
        Operating System :: MacOS
        Operating System :: Microsoft :: Windows
        Operating System :: OS Independent
        Operating System :: POSIX
        Operating System :: Unix
        Programming Language :: Python :: 2
        Programming Language :: Python :: 3
        Topic :: Scientific/Engineering
        Topic :: Software Development
        Topic :: Software Development :: Libraries
        """.splitlines() if len(c.strip()) > 0],
    description='Open CL Python bindings',
    long_description=long_description,
    license='BSD',
    packages=find_packages(),
    platforms=["Windows", "Linux", "Mac OS-X", "Unix", "Solaris"],
    package_data={'opencl': ['*.h']}
)
