from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open('requirements.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

setup(
    name='hmpinn',
    version='0.1',
    author='Mattia Barbiere',
    author_email='mattia.barbiere@epfl.ch',
    description='PINN implementation for harmonic maps',
    long_description='Python package for solving the harmonic maps using Physics-Informed Neural Networks (PINNs).',
    long_description_content_type='text/markdown',
    url='https://github.com/MattiaBarbiere/PINNs_for_harmonic_maps/tree/main',
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
    ],
)
