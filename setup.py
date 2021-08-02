from os import path
from setuptools import setup, find_packages

root_dir = path.abspath(path.dirname(__file__))

with open(path.join(root_dir, 'README.md'), 'r') as in_f:
    long_description = in_f.read()

setup(
    name='micro_orm_gen',
    version='0.1.0',
    description='ORM code generation from config',
    packages=find_packages(),
    entry_points={'console_scripts': ['ormgen=ormgen:main']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Anton Paquin',
    author_email='python@antonpaqu.in',
    url='https://github.com/antonpaquin/micro_orm_gen',
)



