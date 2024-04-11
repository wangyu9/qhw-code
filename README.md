# Fast Quasi-Harmonic Weights for Geometric Data Interpolation.

This repo is a c++ implementation of the paper 


*	[**Fast Quasi-Harmonic Weights for Geometric Data Interpolation**](https://dl.acm.org/doi/abs/10.1145/3450626.3459801).

	Yu Wang and Justin Solomon. _Massachusetts Institute of Technology_. 
	
	_ACM Transactions on Graphics 40(4)_.
	_ACM SIGGRAPH 2021_.
	[OpenAccessPaper](https://dl.acm.org/doi/abs/10.1145/3450626.3459801)

![rep](https://user-images.githubusercontent.com/11622280/128125232-eb8188b3-e034-4f59-960d-b2bc9b9ee57f.jpg)



To clone the repo, run

`git clone --recurse-submodules https://github.com/wangyu9/qhw-code.git`

## Installation

To ease the installation of the dependencies, we provide a docker container. To build the docker container, run

`sudo docker build -t qhw .`

To start the docker container:

`docker run -it qhw`

optionally, you can use  `docker run  -v HOST-MACHINE-FOLDER:CONTAINER-MACHINE-FOLDER  -it qhw` to mount a volume to the docker container. 

Alternatively, you are welcome to install all the dependencies manually by following the Dockerfile. It will lead to better performance since the code always runs slower in the dokcer container (which can be thought of as a light-weight virtual machine).    

## Code Compilation 

(You can skip this step if using our docker container.)

To build the c++ project, either within the container or on your own linux machine: 

`cd qhw-code`

`mkdir build; cd build;`

`cmake ..; make;`

If using our docker container, the compiled executable is located at `/qhw/qhw-code/build/qhw`

## Usage

To compute the weights for the examples that we provided in `./data`, run the following command

`./qhw -e /qhw/qhw-code/data/beast-H  --step_size 0.1 -n  4 --solver adamd --project --verbose 0`

The data file folder can be 
`beast-H,
bunny-H, 
raptor-H, 
elephant-H, 
dragon-H,
tibiman-H`.
The files specify the shape (already meshed with handles) as well as the boundary conditions for the corresponding bone weight problem.

The resulting weights are stored in the file "W.mtx", i.e. in the Matrix Market format (.mtx). 

If the code runs correctly, expect to see the following smoothness energy for the provided examples: 

Example	|	Energy	|
|:--------|:--------|
beast-H	| 111.5	
bunny-H | 10509.6
raptor-H | 34585.9
elephant-H | 644.5
dragon-H | 134.5
tibiman-H | 36136.4

## Contact

Email wangyu9@mit.edu for any question and comment. 

## BibTex
```
@article{wang2021fast,
  title={Fast quasi-harmonic weights for geometric data interpolation},
  author={Wang, Yu and Solomon, Justin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={40},
  number={4},
  pages={1--15},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```
