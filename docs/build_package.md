# https://packaging.python.org/tutorials/packaging-projects/

# Structure
1. Place code in 'src' folder
2. The next level is the package (e.g. 'src/gegravity')
3. Within the package directory, place an __init__ with instructions to import hat ever you want imported when the package is loaded. 
       * The import commands should be structured such that they are relative to the current location (e.g. in 
         src/gegravity/__init__.py, the import is "from .OneSectorGE import *"
4. Optionally, the second level can contain sub directories, which become modules. (e.g. see gme)
    * __init__.py's placed in submodules govern what is imported when that submodule is loaded. For example, is there were
      a submodule 'fakemodule', then you could add "src/gegravity/fakemodule/__init__.py" with content like 
      "from .fakescript import fakeclass". Then, with the package, a user could import "from gegravity.fakemodule import 
      fakeclass". Alternatively, they could use "import gegravity.fakemodule as gef", which would make gef.fakeclass available.
    * To load the contents of the module when the whole package is loaded, import statements should be added to the higher 
      level "src/gegravity/__init__.py" too. For example, add "from .fakemodule.fakescript import *"


# Create and upload package
```
f:
cd Research/gegravity
"E:\python_venvs\package_build\Scripts\activate.bat"
python -m build
python -m twine upload --repository testpypi dist/*
```
# create venv and install package
e:
cd python_venvs
python -m venv gegravity
"E:\python_venvs\gegravity\Scripts\activate.bat"
pip install pandas
pip install scipy
pip install gme
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps gegravity-peter-herman


