# FROM ubuntu:16.04
FROM ubuntu:20.04

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
        # apt-utils \
        ##  module-init-tools \ # removed for 20.04
        # openssh-server \
        curl \
        # libfreetype6-dev \
        # libpng12-dev \
        # libzmq3-dev \
        # pkg-config \
        ##  python2.7 \
        ##  python2.7-dev \
	python \
	python-dev \
	## python-pip \
	# python-tk \
	# python-lxml \
	# python-six \
        # rsync \
        # software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pip
# RUN curl -fsSL -O https://bootstrap.pypa.io/get-pip.py && \
#    python2.7 get-pip.py && \
#    rm get-pip.py

# https://stackoverflow.com/questions/44331836/apt-get-install-tzdata-noninteractive

RUN sudo apt-get update
RUN DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends tzdata


# RUN sudo apt-get update && \
#    sudo apt-get install -y autoconf libtool pkg-config



RUN sudo apt-get update


RUN sudo apt-get --assume-yes install git-all
RUN sudo apt-get --assume-yes install cmake

################ Dependencies for Quasi-Harmonic Weights #################

RUN sudo apt-get --assume-yes purge gcc g++ # remove if there is a diff version.

RUN sudo apt-get --assume-yes install gcc-7 g++-7

# RUN sudo apt-get --assume-yes install gcc g++

