:: Serve live updating local HTTP server (click link in terminal)
pdoc3 --http : src/gegravity
:: Create HTML in "docs" directory for package "gegravity", force overwrite
pdoc3 --html --output-dir docs src/gegravity --force

pdoc3 --html --template-dir "docs/templates" --config latex_math=True --output-dir docs src/gegravity --force




:: Using Regular pdoc
pdoc --docformat "google" ./models

