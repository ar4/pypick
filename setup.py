from setuptools import setup

setup(
    name='pypick',
    version='0.0.1',
    description='Horizon picking',
    url='https://github.com/ar4/pypick',
    author='Alan Richardson',
    author_email='alan@ausargeo.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    packages=['pypick'],
    install_requires=['matplotlib', 'numpy'],
)
