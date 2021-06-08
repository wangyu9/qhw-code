FROM ubuntu:16.04

############## BASIC DEPENDENCIES AND COMMON PKGs #####################

RUN apt-get update && \
      apt-get -y install sudo
RUN sudo apt-get update


# Install wget and build-essential
RUN apt-get update && apt-get install -y \
  build-essential \
  wget \
  sudo \
  apt-utils

# Install vim & nano
# RUN sudo apt-get --assume-yes install vim
RUN sudo apt-get --assume-yes install nano

# More basic dependencies and python 2.7
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        module-init-tools \
        openssh-server \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        # python2.7 \
        # python2.7-dev \
	python \
	python-dev \
	python-pip \
	python-tk \
	python-lxml \
	python-six \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pip
# RUN curl -fsSL -O https://bootstrap.pypa.io/get-pip.py && \
#    python2.7 get-pip.py && \
#    rm get-pip.py

RUN sudo apt-get update && \
    sudo apt-get install -y autoconf libtool pkg-config

RUN sudo pip install --upgrade pip
RUN sudo pip install -U setuptools

# Jupyner and common python packages
RUN sudo pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        Pillow
# sklearn somehow fails.

################ Dependencies for Quasi-Harmonic Weights #################


RUN sudo apt-get --assume-yes install gcc=7.5.0 g++=7.5.0

RUN sudo mkdir qhw

# https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v5.7.1.zip
