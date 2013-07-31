from setuptools import setup, find_packages

# TODO: compile Cython code!

setup(
    name='slicedpy',
    version='0.1',
    packages = find_packages(),
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
