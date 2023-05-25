#!/bin/zsh
find . -name "*.py" \
    -not -path "./lib/**" \
    -not -path "./share/**" \
    -or \( -name "raps" -and -not -path "./raps" -and -not -path "./bin/**" \) \
    > rsync-files.txt
