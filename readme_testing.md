Files for Packaging:
1. pyproject.toml :
The key lines to do with packaging are one to six, where we specify what the basic
dependencies for installing the packages are i.e. setuptools and wheel. We might
see some pyproject.toml lines which specify alternative tools for building. These are
the standard tools that you'll come across the most. The rest of the pyproject.toml is
dedicated to configuring our tooling, like pytest settings. And then further down, we

have the configuration for our linter, which is called black and the import sorting tool
called isort.
Be rest assured, as rarely, we will need to write something like a pyproject.toml file
from scratch. Most of the time, we will be using a template, pulling and modifying a
pyproject.toml from another project, or generated with the tool. The same applies to
the setup file, manifest.in, and the tooling configurations. So, we don’t have to write
a file like this line by line.


2. setup.py
The majority of the packaging functionality resides inside the Setup file. You can see
the package metadata. We're going to call the package ‘titanic_model’. The setup
file contains metadata about the package, such as its description, author, and
compatibility with a specific version of Python. The version of the package is
determined by reading a version file, which contains a single value specifying the
version. This version value is then assigned to the metadata dictionary in the Setup
file. The requirements for the package are also specified, usually by using a list of
requirements from a requirements directory. All these values are passed as
arguments to the setup function from the setuptools library, which is crucial for
creating the package.



3. Manifest.in
The Manifest.in file is used to specify the files that should be included or excluded in
the package. This file is responsible for ensuring that important files like the pickle
file, as well as the train and test CSVs, are included in the package. The inclusion of
the CSVs allows other applications that depend on the package to use them for
testing purposes.



4. mypy.ini:
The mypy.ini file is used to configure the specific type hints that we want to focus on
and pay attention to in our code. However, this is manageable and effortless.

Tooling: Tools that are used to manage our package: In addition to our pytest library,
which we're using for testing, we may have a few other libraries viz. black, flake, mypy,
isort.
• black is a code-styling enforcement.
• flake8 is a linting tool to tell us where we are not adhering to good Python
conventions.
• mypy is a type-checking tool and
• isort is a tool for ensuring our imports are in the correct order.

The prediction of the test set and calculation of accuracy previously kept inside the
train_pipeline.py is shifted to the test_predictions.py file.