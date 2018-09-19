from setuptools import setup

setup(name='pipefinch',
      version='0.1',
      description='Spike sorting pipeline based on intan recording system and mountainsort',
      url='http://github.com/zekearneodo/pipefinch',
      author='Zeke Arneodo',
      author_email='ezequiel@ini.ethz.ch',
      license='MIT',
      packages=['pipefinch'],
      install_requires=['numpy',
                        'pandas>=0.23',
                        'mountainlab_pytools',
                       ],
      zip_safe=False)