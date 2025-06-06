from setuptools import setup, find_packages

setup(
    name='hmpinn',       # Replace with your package name 
    version='0.1',                  # Package version
    author='Philipp Weder',             # Author name
    author_email='philipp.weder@epfl.ch',  # Author email
    description='PINN implementation for harmonic maps',  # Short description
    long_description='TBD',  # Full description
    long_description_content_type='text/markdown',  # Description content type
    url='https://github.com/weder/hmpinnß',  # URL to your package repository
    packages=find_packages(),       # Automatically find all packages in the directory
    install_requires=[              # List of dependencies required to run your package
        # Add more dependencies as needed
    ],
    classifiers=[                   # Classifiers to categorize your package (optional)
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        # Add more classifiers as needed
    ],
)
