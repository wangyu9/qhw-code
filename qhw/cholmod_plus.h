#include "cholmod.h"

double my_cholmod_l_norm_dense //getEntryRef
(
	/* ---- input ---- */
	cholmod_dense* X,	/* matrix to compute the norm of */
	int norm,		/* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
	/* --------------- */
	cholmod_common* Common
);

double my_cholmod_l_norm_sparse
(
	/* ---- input ---- */
	cholmod_sparse* A,	/* matrix to compute the norm of */
	int norm,		/* type of norm: 0: inf. norm, 1: 1-norm */
	/* --------------- */
	cholmod_common* Common
);

double my_cholmod_l_pnorm_sparse //getEntryRef
(
	/* ---- input ---- */
	cholmod_sparse* X,	/* matrix to compute the norm of */
	int power,
	/* --------------- */
	cholmod_common* Common
);

//cholmod_sparse* my_cholmod_l_ssmult
//(
//	/* ---- input ---- */
//	cholmod_sparse* A,	/* left matrix to multiply */
//	cholmod_sparse* B,	/* right matrix to multiply */
//	int stype,		/* requested stype of C */
//	int values,		/* TRUE: do numerical values, FALSE: pattern only */
//	int sorted,		/* if TRUE then return C with sorted columns */
//	/* --------------- */
//	cholmod_common* Common
//);

static void r_cholmod_sdmult_mkl
(
	/* ---- input ---- */
	cholmod_sparse* A,	/* sparse matrix to multiply */
	int transpose,	/* use A if 0, or A' otherwise */
	double alpha[2],   /* scale factor for A */
	double beta[2],    /* scale factor for Y */
	cholmod_dense* X,	/* dense matrix to multiply */
	/* ---- in/out --- */
	cholmod_dense* Y,	/* resulting dense matrix */
	/* -- workspace -- */
	double* W		/* size 4*nx if needed, twice that for c/zomplex case */
);

int my_cholmod_l_sdmult_mkl
(
	/* ---- input ---- */
	cholmod_sparse* A,	/* sparse matrix to multiply */
	int transpose,	/* use A if 0, otherwise use A' */
	double alpha[2],   /* scale factor for A */
	double beta[2],    /* scale factor for Y */
	cholmod_dense* X,	/* dense matrix to multiply */
	/* ---- in/out --- */
	cholmod_dense* Y,	/* resulting dense matrix */
	/* --------------- */
	cholmod_common* Common
);

// New: 

void my_cholmod_l_ddmult2
(
	/* ---- input ---- */
	cholmod_dense* X,
	cholmod_dense* Y,
	/* ---- output ---- */
	cholmod_dense* R
);

void my_cholmod_l_sparse_tensor_diag_mul2
(
	cholmod_sparse* A,
	double* h,
	cholmod_sparse* B,
	int option
);

void my_cholmod_l_sparse_diag_mul2
(
	cholmod_sparse* A,
	double* h, 
	cholmod_sparse* B
);

void my_cholmod_l_sparse_diag_mul2_same_pattern
(
	cholmod_sparse** mA,
	double** mb,
	cholmod_sparse** mB,
	int num
);

void my_cholmod_l_check_same_pattern
(
	cholmod_sparse* A,
	cholmod_sparse* B
);

void my_cholmod_l_assign_same_pattern
(
	cholmod_sparse* A, // src matrix.
	cholmod_sparse* B // dest matrix. 
);

int my_cholmod_l_assmble_alap
(
	cholmod_sparse* GxT,
	cholmod_sparse* GyT,
	cholmod_sparse* GzT, // use for only dim==3
	double** h, // h[0-2] or h[0-5] stores the 2x2 or 3x3 tensors.
	int dim,
	bool lower_only,
	/* --------------- */
	int*& II,
	int*& JJ,
	double*& VV
	/* --------------- */
	// cholmod_common* Common
);

int my_cholmod_l_assmble_alap_off_diag
(
	cholmod_sparse* GxT1,
	cholmod_sparse* GyT1,
	cholmod_sparse* GzT1, // use for only dim==3
	cholmod_sparse* GxT2,
	cholmod_sparse* GyT2,
	cholmod_sparse* GzT2, // use for only dim==3
	double** h, // h[0-2] or h[0-5] stores the 2x2 or 3x3 tensors.  
	int dim,
	/* --------------- */
	int*& II,
	int*& JJ,
	double*& VV
	/* --------------- */
	// cholmod_common* Common
);

void my_cholmod_l_sum_same_pattern
(
	cholmod_sparse** mA,
	cholmod_sparse* B,
	int num_src
);

void my_cholmod_l_sparse_mul2_same_pattern
(
	cholmod_sparse** mA, // mA[0~num_src-1] stacks matrices 
	cholmod_sparse** mB, // mB[0~num_src-1] stacks matrices 
	cholmod_sparse** mR, // mR stacks results that mR[i] = mA[i] * mB[i].
	int num_src
);

void my_cholmod_l_print
(
	cholmod_sparse* A
);