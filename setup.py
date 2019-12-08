import matplotlib.pyplot as plt
import datetime
import sys

from setuptools import find_packages
from setuptools import setup

nightly = False
if '--nightly' in sys.argv:
    nightly = True
    sys.argv.remove('--nightly')

project_name = 'tensorflow-docs'
version = '0.0.0'
if nightly:
    project_name = 'tfds-nightly'
    datestring = datetime.datetime.now().strftime('%Y%m%d%H%M')
    version = '%s-dev%s' % (version, datestring)

DOCLINES = __doc__.split('\n')[0]

REQUIRED_PKGS = [
    'astor',
    'absl-py',
    'six', 'matplotlib', 'pandas', 'scikit-learn', 'numpy', 'keras'
]

TESTS_REQUIRE = [
    'jupyter',
]

if sys.version_info.major == 3:
    # Packages only for Python 3
    pass
else:
    # Packages only for Python 2
    TESTS_REQUIRE.append('mock')
    REQUIRED_PKGS.append('futures')  # concurrent.futures

if sys.version_info < (3, 4):
    # enum introduced in Python 3.4
    REQUIRED_PKGS.append('enum34')

setup(
    name=project_name,
    version=version,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    url='http://github.com/tensorflow/docs',
    download_url='https://github.com/tensorflow/docs/tags',
    license='Apache 2.0',
    packages=find_packages('tools'),
    package_dir={'': 'tools'},
    scripts=[],
    install_requires=REQUIRED_PKGS,
    extras_require={
        'tests': TESTS_REQUIRE,
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow api reference',
)
