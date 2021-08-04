
#include "cholmod.h"
// #include <Include/cholmod.h>

// #include "cholmod_function.h"

int initLHS(cholmod_sparse* A, cholmod_common* cm); 

int initRHS(cholmod_dense* B); 

int linSolve_ori(cholmod_sparse* A, cholmod_dense* B, cholmod_dense* XXX, cholmod_common* cm); // This is the one from cholmod_l_example. 

int linSolve(cholmod_sparse* A, cholmod_dense* B, cholmod_dense* X, cholmod_common* cm); // This is the one modified from MATLAB wrapper. 

int symbolic_factorize(cholmod_sparse* A, cholmod_factor*& L, cholmod_common* cm);

int numerical_factorize(cholmod_sparse* A, cholmod_factor* L, cholmod_common* cm);

int solveLLT(cholmod_factor* L, cholmod_dense* B, cholmod_dense* XXX, cholmod_common* cm);