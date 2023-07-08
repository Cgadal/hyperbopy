from setuptools import setup, find_packages

setup(name="hyperbopy",
      packages=find_packages(),
      python_requires='>=3',
      install_requires=[
          "numpy", "matplotlib"],
      author='Cyril Gadal',
      license='Apache-2.0',
      version='0.1.2',
      zip_safe=False,
      )
