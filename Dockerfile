# FROM ubuntu:16.04
FROM ubuntu:20.04
# it is much easiler to install mkl in command lines for ubuntu 20. otherwise you may have to manually install it from intel.com

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



ENV CC=/usr/bin/gcc-7
ENV CXX=/usr/bin/g++-7

RUN ln -s /usr/bin/gcc-7 cc
RUN ln -s /usr/bin/gcc-7 gcc
RUN ln -s /usr/bin/g++-7 c++
RUN ln -s /usr/bin/g++-7 g++

RUN sudo apt-get --assume-yes install libgmp3-dev # otherwise I am missing the file gmp.h
RUN sudo apt-get --assume-yes install libmpfr-dev # for mpfr.h 


RUN sudo mkdir qhw

RUN sudo apt-get --assume-yes install libblas-dev liblapack-dev 

RUN sudo apt-get --assume-yes install m4 # GraphBLAS requires m4

# RUN sudo apt-get --assume-yes install libopenblas-dev

RUN sudo mkdir /qhw/external

Run wget -q -O - "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v5.7.1.tar.gz" | tar -xzf - -C /qhw/external

RUN sudo mkdir /qhw/external/LibSuiteSparse-5.7.1


RUN bash -c "cd /qhw/external/SuiteSparse-5.7.1/ && make config LAPACK=-llapack BLAS=-lblas   INSTALL=/qhw/external/LibSuiteSparse-5.7.1 && make library  LAPACK=-llapack BLAS=-lblas  INSTALL=/qhw/external/LibSuiteSparse-5.7.1"

RUN sudo mkdir /qhw/external/tensorflow/

RUN sudo mkdir /qhw/external/tensorflow/2.4.0

Run wget -q -O - "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz" |  tar -xzf - -C /qhw/external/tensorflow/2.4.0


# install armadillo following: http://arma.sourceforge.net/download.html

RUN sudo apt-get --assume-yes install libopenblas-dev  liblapack-dev  libarpack2-dev  libsuperlu-dev

RUN sudo mkdir /qhw/external/armadillo

RUN  wget -q -O - "http://sourceforge.net/projects/arma/files/armadillo-10.6.1.tar.xz"  |  tar -xJf - -C /qhw/external/armadillo

RUN  bash -c "cd /qhw/external/armadillo/armadillo-10.6.1 && ./configure && make && sudo make install"
# following http://codingadventures.org/2020/05/24/how-to-install-armadillo-library-in-ubuntu/

# intel MKL

RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --force-yes --no-install-recommends  libmkl-full-dev

RUN sudo apt-get --assume-yes install gdb

RUN  bash -c "cd /qhw && git clone git clone --recurse-submodules https://github.com/wangyu9/qhw-code.git"
RUN  bash -c "cd /qhw/qhw-code && mkdir build && cd build && cmake .. && make"

# COPY ./linsolve /qhw/qhw
