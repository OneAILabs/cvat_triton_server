from setuptools import setup, find_packages

setup(
    name='webapp',
    version='1.0',
    long_description=__doc__,
    packages=['webapp'],
    package_dir={'webapp': '.'},
    package_data={'webapp': ['*.py', './src/*.py', './v1/*.py', './v1/api/*.py']},
    include_package_data=True,
    zip_safe=False,
    install_requires=['Flask']
)
