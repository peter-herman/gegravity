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
5. Create/update setup.cfg, which should sit in the same directory as src. 

# Create and upload package (see https://packaging.python.org/en/latest/tutorials/packaging-projects/)
To test, use the test pypi server. In setup.cfg, replace package name with "name = gegravity-peter-herman". API access is handled by API key in config file at $HOME/.pypirc (may need to recreate if it is a new system)
```
f:
cd Research/gegravity
"E:\python_venvs\package_build\Scripts\activate.bat"
python -m build
python -m twine upload --repository testpypi dist/*
```
The test package can be viewed at https://test.pypi.org/project/gegravity-peter-herman
```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps gegravity-peter-herman
```



When ready to upload to pypi, return package name to just the name then:

```
python -m build
python -m twine upload dist/*
```


# create venv and install package
```
e:
cd python_venvs
python -m venv gegravity
"E:\python_venvs\gegravity\Scripts\activate.bat"
pip install pandas
pip install scipy
pip install gme
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps gegravity-peter-herman
```

Or
```
conda create --prefix file_path\venv_name python=3.8
pip install gme==1.3
```
