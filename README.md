# Fast quasi-harmonic weights for geometric data interpolation.

This repo is a c++ implementation of the paper 

*	**Fast Quasi-Harmonic Weights for Geometric Data Interpolation**.

	Yu Wang and Justin Solomon.
	_ACM Transactions on Graphics 40(4)_.
	_ACM SIGGRAPH 2021_.
	[OpenAccessPaper](https://dl.acm.org/doi/abs/10.1145/3450626.3459801)

To clone the repo, run

`git clone --recurse-submodules git@github.com:wangyu9/qhw-code.git`

To ease the installation of the dependencies, we provide a docker container. To build the docker container, run

`sudo docker build -t qhw .`

To build the c++ project, 

`cd qhw`
`mkdir build; cd build;`
`cmake ..; make;`

To compute the weights for the examples given in ./data, run the following commands

`./qhw -e /qhw/qhw/data/beast-H  --step_size 0.1 -n  4 --solver adamd --project --verbose 0`

The data file folder can be 
beast-H
bunny-H 
raptor-H 
elephant-H 
dragon-H
tibiman-H

The resulting weights are stored in the file "W.mtx". 
