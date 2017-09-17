from distutils.core import setup

NAME = 'LibML'
AUTHOR = 'Austen Schunk'
EMAIL = 'aschunk4@gmail.com'
DESCRIPTION = 'A collection of modules in python that can be used for machine learning'
VERSION = '0.1.03'
PACKAGES = ['libml', 'libml.Classification', 'libml.Utilities']
LICENSE = 'MIT License'
URL = 'https://pypi.python.org/packages/c3/10/3a58547058f5197bc19a9d252b9a68368d6707d52dac058f263c7201c294/LibML-0.1.03.tar.gz'

setup(
	name = NAME,
	author = AUTHOR,
	author_email = EMAIL, 
	version = VERSION,
	description = DESCRIPTION,
	packages = PACKAGES,
	license = LICENSE,
	url = URL,
	)
