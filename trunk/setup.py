from setuptools import setup, find_packages

setup(name='dmath',
      version='0.9',
      description="Math routines for Python's Decimal type",
      author='Brian Beck, Christopher Hesse',
      author_email='exogen@gmail.com, christopher.hesse@gmail.com',
      url='http://code.google.com/p/dmath',
      download_url='http://dmath.googlecode.com/svn/trunk/',
      packages=find_packages(),
      license='MIT',
      keywords='decimal math precision trigonometry trigonometric',
      classifiers=["Development Status :: 4 - Beta",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: Software Development :: Libraries :: Python Modules"
                   ],
     )
