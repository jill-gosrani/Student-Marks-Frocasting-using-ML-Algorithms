## Creating My First End to End Machine Learning Project.
#Steps done
1. Created a README.md file
2. Created virtual environment using venv package `python -m venv /path/to/new/virtual/environment`
3. Activated the virtual environment using command `./Script/Activate.ps1`  ./ is to identify the script
4. Setup the connections with git repository "https://github.com/jill-gosrani/MLProject.git"
5. Created a gitignore file - The `.gitignore` file specifies patterns of files and directories that should be ignored by Git, preventing them from being tracked or committed. 
6. Created 2 new files:
    i) setup.py
    ii)requirements.txt
7. Created a new folder "src" and the __init__ constructor

## Setup.py
Responsible for creating this Machine Learning Application as a package, which can be deployed on `https://pypi.org/` and used by others.

We will use `setuptools`package and import `find_packages()`. `find_packages()` will automatically find out all the package that available in the entire application and those that we have mentioned in the requirements.txt file.

`Setuptools` is a collection of enhancements to the Python `distutils` that allow developers to more easily build and distribute Python packages, especially ones that have dependencies on other packages.Packages built and distributed using setuptools look to the user like ordinary Python packages based on the `distutils`.

`setup()` will basically be the metadata information about the entire project. 
Important parameters in `setup()` are:
    1) packages = `packages` parameter is used to specify the Python packages that should be included when distributing and installing your project.

    2) install_requires = `install_requires`  parameter is used to specify the dependencies of the package. It is a list of strings that define the required packages and their versions.


## Requirements.txt
This file will have all the packages that we need to install while implementing the project.

## __init__.py

Whenever the `find_packages` will run in `setup.py` it will go through the entire project and look for __init__ constructor.

the `find_packages()` function is a utility method used to automatically discover and collect all Python packages (directories containing an `__init__.py` file) within a specified directory.

The `find_packages()` function simplifies the process of listing and including packages in your project's setup script by automatically finding and including all the packages without the need for manual listing.

When `find_packages()` is called, it recursively searches for packages starting from the specified directory and includes them in the package list. It returns a list of all the packages found.

Here's a brief explanation of how `find_packages()` works:

1) The function takes an optional where parameter, which specifies the root directory to start the search for packages. If not provided, it defaults to the current directory.

2) It scans the directory and its subdirectories recursively to find all directories that contain an `__init__.py` file, indicating a Python package.

3) Each discovered package is added to the list of packages.

4) The function returns the list of packages found.

## Problem Statement