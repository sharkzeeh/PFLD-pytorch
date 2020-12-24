#!/bin/bash

NEEDED_COMMANDS="docker"

for cmd in ${NEEDED_COMMANDS} ; do
    if ! command -v ${cmd} &> /dev/null
    then
        echo "Please install ${cmd}!"
        exit 1
    fi
done