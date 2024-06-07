::HOW TO:
:: Run one of these commands from a terminal with in which a version of python/conda is active and pdoc3 installed.
:: To get the live updating version to ignore the src directory, change directory befor erunning the command.__
:: To get the index page to include the manually written documentation:
::      Open the __init__.py file that contains the documentation (src/gegravity/__init__.py)
::      Move the import statements after the multiline string containing the documentation
::      After the html pages are completed, move the import statements back up.


:: Serve live updating local HTTP server (click link in terminal)
pdoc3 --http : gegravity


:: Or, more recently, had to use the full path to get to pdoc3 (06/07/2024). The HTTP version was hosting an old version of the package (maybe the public version of gegravity 0.2?)
c:\users\peter.herman\appdata\roaming\python\python311\scripts\pdoc3.exe --html --output-dir docs src/gegravity --force
c:\users\peter.herman\appdata\roaming\python\python311\scripts\pdoc3.exe --http : gegravity

:: Create HTML in "docs" directory for package "gegravity", force overwrite
:: IN ORDER TO RENDER DOCUMENTATION ON INDEX: In order to get the documentation text that appears in __
pdoc3 --html --output-dir docs src/gegravity --force

pdoc3 --html --template-dir "docs/templates" --config latex_math=True --output-dir docs src/gegravity --force




:: Using Regular pdoc
pdoc --docformat "google" ./models

