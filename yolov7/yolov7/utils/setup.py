from setuptools import setup, find_packages

setup(
    name='utils',
    version='1.0',
    long_description=__doc__,
    packages=['utils'],
    package_dir={'utils': '.'},
    package_data={'utils': ['*.py']},
    include_package_data=True,
    zip_safe=False
)
