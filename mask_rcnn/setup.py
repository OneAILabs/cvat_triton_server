from setuptools import setup, find_packages

setup(
    name='cvat_custom',
    version='1.0',
    long_description=__doc__,
    packages=['cvat_custom'],
    package_dir={'cvat_custom': '.'},
    package_data={'cvat_custom': ['*.py', './mrcnn/*.py']},
    include_package_data=True,
    zip_safe=False
)
