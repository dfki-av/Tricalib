##############################################################################
# Developed at DFKI GmbH 2024-25
# Written by Rahul Jakkamsetty <rahul.jakkamsetty@dfki.de>, December 2024
##############################################################################

from setuptools import setup, find_packages
from os import path


ext_modules = []


# Parse requirements.txt
with open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')
install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
install_requires = [x for x in install_requires if '--extra' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]


# Call the setup.py
setup(
    name='tricalib',
    version=0.4,
    author='Rahul Jakkamsetty',
    license='internal',
    packages=find_packages(exclude=['docs', 'libs', 'test*', 'examples', 'debug', 'data']),
    install_requires=install_requires,
    dependency_links=dependency_links,
    ext_modules=ext_modules,
)
