import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eiot",
    version="1.0.0",
    author="Sal Garcia",
    author_email="salvadorgarciamunoz@gmail.com",
    description="A Python and Matlab implementation of Extended Iterative Optimization Technology (EIOT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salvadorgarciamunoz/eiot",
    py_modules= ["eiot", "eiot_extras"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pyphimva",
        "numpy",
        "matplotlib",
        "scipy",
        "pyomo",
        "pandas",
        "bokeh",
    ],
)
