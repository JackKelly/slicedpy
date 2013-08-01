from setuptools import setup, find_packages, Extension
from os.path import join

CYTHON_DIR = 'cython'

try:
    # This trick adapted from 
    # http://stackoverflow.com/a/4515279/732596
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

if use_cython:
    sources = [join(CYTHON_DIR, 'feature_detectors.pyx')]
    extensions = [Extension("slicedpy.feature_detectors", sources=sources)]
    ext_modules = cythonize(extensions)
else:
    ext_modules = [
        Extension("slicedpy.feature_detectors", 
                  [join(CYTHON_DIR, 'feature_detectors.c')]),
    ]


setup(
    name='slicedpy',
    version='0.1',
    packages = find_packages(),
    ext_modules = ext_modules,
    install_requires = ['numpy', 'pandas', 'pda'],
    description='Estimate the energy consumed by individual appliances from whole-house power meter readings',
    author='Jack Kelly',
    author_email='jack@jack-kelly.com',
    url='https://github.com/JackKelly/slicedpy',
    download_url = "https://github.com/JackKelly/slicedpy/tarball/master#egg=slicedpy-dev",
    long_description=open('README.md').read(),
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='smartmeters power electricity energy analytics redd disaggregation nilm nialm'
)
