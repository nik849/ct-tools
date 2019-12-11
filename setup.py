import setuptools

setuptools.setup(
    name="cttools",
    version="0.0.1",
    author="nik849",
    author_email="nickhale@protonmail.ch",
    description="Tomography Toolkit in Pure Python",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/nik849/cttools",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'scikit-image',
            'natsort',
            'imageio',
            'nose',
            'ray'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Mac/Linux"
    ],
)
