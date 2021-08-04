# Fast quasi-harmonic weights for geometric data interpolation.

This repo is a c++ implementation of the paper 

*	**Fast Quasi-Harmonic Weights for Geometric Data Interpolation**.

	Yu Wang and Justin Solomon.
	_ACM Transactions on Graphics 40(4)_.
	_ACM SIGGRAPH 2021_.
	[OpenAccessPaper](https://dl.acm.org/doi/abs/10.1145/3450626.3459801)

`sudo docker build -t qhw .`


`git clone --recurse-submodules git@github.com:wangyu9/qhw-code.git`


`./qhw -e /qhw/qhw/data/raptor-H  --step_size 0.1 -n  4 --solver adamd --project --verbose 0`
