
#include "cholmod.h"
// #include <Include/cholmod.h>

// #include "cholmod_function.h"


/* This one is modified from Tim Davis's Demo/cholmod_demo.h  */

#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#define TRUE 1
#define FALSE 0

#define CPUTIME (SuiteSparse_time ( ))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(a)   (((a) >= (0)) ? (a) : -(a))

#include "cholmod_function.h"


int initLHS(cholmod_sparse* A, cholmod_common* cm); 

int initRHS(cholmod_dense* B); 

int linSolve_ori(cholmod_sparse* A, cholmod_dense* B, cholmod_dense* XXX, cholmod_common* cm); // This is the one from cholmod_l_example. 

int linSolve(cholmod_sparse* A, cholmod_dense* B, cholmod_dense* X, cholmod_common* cm); // This is the one modified from MATLAB wrapper. 

int symbolic_factorize(cholmod_sparse* A, cholmod_factor*& L, cholmod_common* cm);

int numerical_factorize(cholmod_sparse* A, cholmod_factor* L, cholmod_common* cm);

int solveLLT(cholmod_factor* L, cholmod_dense* B, cholmod_dense* XXX, cholmod_common* cm);