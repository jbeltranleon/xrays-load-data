from setuptools import setup

setup(name='xrays-load-data',
      version='0.1',
      description='Load the xrays iamges',
      url='http://github.com/jbeltranleon/xrays-load-data',
      author='Jhon Beltran',
      author_email='jbeltranleon@gmail.com',
      license='MIT',
      packages=['xrays-load-data'],
      zip_safe=False,
      install_requires=[
       "numpy",
       "pandas",
       "glob",
       "os",
       "sklearn"
   ])