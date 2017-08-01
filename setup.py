from setuptools import setup

setup(
    name='yabox',
    version='1.0.1',
    description='Yet another black-box optimization library for Python',
    author='Pablo Rodriguez-Mier',
    author_email="pablo.rodriguez.mier@gmail.com",
    packages=['yabox', 'yabox.algorithms', 'yabox.problems'],
    install_requires=['numpy>=1.7'],
    extras_require={
        'notebooks': ['jupyter>=1.0.0'],
        'visualization': ['matplotlib>=1.3'],
        'show_progress': ['tqdm>=4.0.0']},
    tests_require=['pytest', 'pytest-pep8'],
    url='https://github.com/pablormier/yabox',
    download_url='https://github.com/pablormier/yabox/archive/1.0.1.tar.gz',
    keywords=['optimization', 'black-box', 'data science', 'evolutionary', 'algorithms'],
    license='Apache License 2.0',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'License :: OSI Approved :: Apache Software License',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 3.4'],
)
