import setuptools

setuptools.setup(
    name='tsfewshot',
    description='Few-shot learning for time series',
    author='Anonymized',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='fewshot timeseries',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'sklearn', 'numpy', 'pandas', 'torch', 'tqdm', 'matplotlib', 'higher'
    ],
    python_requires='>=3.8',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
)
