import setuptools

setuptools.setup(name='SMLM_analysis',
      version='0.1',
      description='Collection of Python files for processing SMLM data',
      url='http://github.com/drmatthews/SMLM_analysis',
      author='Dan Matthews',
      author_email='dr.dan.matthews@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(exclude=['docs', 'test_data']),     
      zip_safe=False)