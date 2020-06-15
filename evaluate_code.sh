#!/bin/bash

app_name=$CI_COMMIT_REF_NAME

cat /proc/version

echo "Installing curl"
apt install -y curl

echo "Updating apt-get"
apt-get update

echo "Updating python"
apt -y install python3-setuptools python3-pip
apt-get -y install python3

echo "Updating pip3"
pip3 install --upgrade pip

echo "Updating libgcc"
conda update libgcc

echo "Installing unixodbc"
apt-get -y install unixodbc unixodbc-dev

# Required to install for pyodbc
echo "Installing build-essential"
apt-get -y install --reinstall build-essential

if [ -d "./tests" ]
then
    if [ ! -f "./requirements.txt" ]
    then
        echo "No requirements.txt file found"
        exit 1
    fi

    pip3 install -r requirements.txt

    echo "import pytest" >> ./run_tests.py
    echo 'pytest.main(["-x", "./tests", "-vv"])' >> ./run_tests.py
    python3 ./run_tests.py

    if [ "$?" -gt "0" ]
    then
        echo "Test failed"
        exit 1
    fi

else
    echo "No tests"
fi
