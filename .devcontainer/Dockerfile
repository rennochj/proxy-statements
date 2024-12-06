
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

RUN pip install --upgrade pip
RUN pip install poetry==1.7.1

RUN <<EOT
    # pip3 config --global set global.index https://cgrepo.capgroup.com/repository/cgpypi/pypi
    # pip3 config --global set global.index-url https://cgrepo.capgroup.com/repository/cgpypi/simple
    # pip3 config --global set global.trusted-host cgrepo.capgroup.com
    pip3 install setuptools --upgrade
EOT

# ENV GIT_SSL_CAINFO="/cacert.pem"
