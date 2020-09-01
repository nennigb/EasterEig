#!/bin/bash

# Script to publish releases on pypi or on test.pypi

for i in "$@"
do
    case "$i" in
    -i|--install) pip install twine
    ;;
    -t|--test)  echo "> Deploy to test.pypi..."
    rm dist/eastereig*.whl
    pip wheel . -w dist    
    twine upload --repository testpypi dist/eastereig*.whl
    # to test install from test.pypi, run
    # pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple eastereig
        ;;
    -d|--deploy) echo "> Deploy to pypi..."
    rm dist/eastereig*.whl
    pip wheel . -w dist    
    twine upload dist/eastereig*.whl
        ;;
    esac
done

