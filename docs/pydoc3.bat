:: Serve live updating local HTTP server (click link in terminal)
pdoc3 --http : models
:: Create HTML in "docs" directory for package "models", force overwrite
pdoc3 --html --output-dir docs models --force

pdoc3 --html --template-dir "docs/templates" --config latex_math=True --output-dir docs models --force
pdoc3 --template-dir "docs/templates" --http : models