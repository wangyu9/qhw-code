#include "cholmod_internal.h"
#include "cholmod_matrixops.h"

#include "cholmod_plus.h"

// #include "cholmod_demo.h"
#include "cholmod_l.h"
#include "cholmod_plus.h"

#define RAISE_ERROR(msg)  fprintf(stderr, (msg)); // wangyu: my own ERROR reporter. 

#undef ERROR
#define ERROR(status,msg) fprintf(stderr, (msg));printf("Error: %d, %s", status, msg)


#include <assert.h>
void my_assert(bool condition) {
	if (!condition) {
		assert(false);
		printf("Assertion failed! Something is wrong.\n");
	}
}
#undef ASSERT
// #define ASSERT(expression) assert((expression))

void ASSERT(bool condition) {
	if (!condition) {
		assert(false);
		printf("Assertion failed! Something is wrong.\n");
	}
	assert(condition);
}

// #define ASSERT(expression) 	if (!((expression))) printf("Assertion failed! Something is wrong.\n");assert((expression))
// #define ASSERT(expression) 	my_assert((expression)), assert((expression))
// the \ character must be the last character on the line. 
// (even if it is just white space afterward) 

#undef RETURN_IF_NULL_COMMON
#define RETURN_IF_NULL_COMMON(xxx) ; 
// wangyu: do noting for RETURN_IF_NULL_COMMON
// Here RETURN_IF_NULL_COMMON always return for me somehow. 

// getEntryRef is  modified from cholmod_norm_dense from cholmod_norm.c

// wangyu: *redefine Int here is critical*, since simply #include "cholmod_internal.h" 
// in this way will not set it right for cholmod_l_xx. 
// Otherwise my_norm etc will return *wrong* results. 
#undef Int
#define Int SuiteSparse_long


// Not to really include, but use it to jump to the files of interests. 
#if 0
#include "../MatrixOps/cholmod_norm.c"
#include "Check/cholmod_write.c"
#include "MATLAB/cholmod2.c"
#include "cholmod_matlab.c"
#include "cholmod2.c"
#include "ldlsolve.c"
#include "Cholesky/cholmod_factorize.c"
#include "Cholesky/cholmod_analyze.c"
#include "Cholesky/cholmod_solve.c"
#include "Core/cholmod_memory.c"
#include "Core/cholmod_sparse.c"
#include "Core/cholmod_transpose.c"
#include "MatrixOps/cholmod_norm.c"
#include "MatrixOps/cholmod_ssmult.c"
#include "MatrixOps/cholmod_sdmult.c"
#include "MatrixOps/t_cholmod_sdmult.c"
#endif

/********************************* cholmod_norm.c **************************************/

double* getEntryPtr //
(
	/* ---- input ---- */
	cholmod_dense* X,	/* matrix to compute the norm of */
	const int row, const int col,
	/* --------------- */
	cholmod_common* Common
)
{
	double* Xx, * Xz;
	Int nrow, ncol, i, j, p, d, xtype;

	double* ptr = NULL;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	RETURN_IF_NULL_COMMON(NULL);
	RETURN_IF_NULL(X, NULL);
	RETURN_IF_XTYPE_INVALID(X, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, NULL);
	Common->status = CHOLMOD_OK;
	ncol = X->ncol;
	nrow = X->nrow;
	if (row < 0 || col < 0 || row >= nrow || col >= ncol)
	{
		ERROR(CHOLMOD_INVALID, "invalid row and col pair");
		return (NULL);
	}

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	
	d = X->d;
	Xx = (double*)X->x; // wangyu:  Xx = X->x;
	Xz = (double*)X->z; // wangyu:  Xz = X->z;
	xtype = X->xtype;

	i = row;
	j = col;
	p = i + j * d;
	{
		switch (xtype)
		{

		case CHOLMOD_REAL:
			ptr = &(Xx[p]); 
			//	x = Xx[i];
			break;

		case CHOLMOD_COMPLEX:
			ERROR(CHOLMOD_INVALID, "invalid row and col pair");
			return (NULL);
			//	x = Xx[2 * i];
			// z = Xx[2 * i + 1];
			break;

		case CHOLMOD_ZOMPLEX:
			ERROR(CHOLMOD_INVALID, "invalid row and col pair");
			return (NULL);
			//	x = Xx[i];
			//	z = Xz[i];
			break;
		}
	}

	return ptr; 
}


// below is slightly modified from cholmod_norm_dense from cholmod_norm.c

static double abs_value
(
	int xtype,
	double* Ax,
	double* Az,
	Int p,
	cholmod_common* Common
)
{
	double s = 0;
	switch (xtype)
	{
	case CHOLMOD_PATTERN:
		s = 1;
		break;

	case CHOLMOD_REAL:
		s = fabs(Ax[p]);
		break;

	case CHOLMOD_COMPLEX:
		s = SuiteSparse_config.hypot_func(Ax[2 * p], Ax[2 * p + 1]);
		break;

	case CHOLMOD_ZOMPLEX:
		s = SuiteSparse_config.hypot_func(Ax[p], Az[p]);
		break;
	}
	return (s);
}


double my_cholmod_l_norm_dense //getEntryRef
(
	/* ---- input ---- */
	cholmod_dense* X,	/* matrix to compute the norm of */
	int norm,		/* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
	/* --------------- */
	cholmod_common* Common
)
{
	double xnorm, s, x, z;
	double* Xx, * Xz, * W;
	Int nrow, ncol, d, i, j, use_workspace, xtype;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	//RETURN_IF_NULL_COMMON(EMPTY); // wangyu remote this one or it will return. 
	RETURN_IF_NULL(X, EMPTY);
	RETURN_IF_XTYPE_INVALID(X, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, EMPTY);
	Common->status = CHOLMOD_OK;
	ncol = X->ncol;
	if (norm < 0 || norm > 2 ) // wangyu remove || (norm == 2 && ncol > 1)
	{
		ERROR(CHOLMOD_INVALID, "invalid norm");
		return (EMPTY);
	}

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	nrow = X->nrow;
	d = X->d;
	Xx = (double*)X->x; // wangyu:  Xx = X->x;
	Xz = (double*)X->z; // wangyu:  Xz = X->z;
	xtype = X->xtype;

	/* ---------------------------------------------------------------------- */
	/* allocate workspace, if needed */
	/* ---------------------------------------------------------------------- */

	W = NULL;
#if 0
	use_workspace = (norm == 0 && ncol > 4);
	if (use_workspace)
	{
		CHOLMOD(allocate_work) (0, 0, nrow, Common);
		W = (double*)Common->Xwork; // wangyu: W = Common->Xwork;
		if (Common->status < CHOLMOD_OK)
		{
			/* oops, no workspace */
			use_workspace = FALSE;
		}
	}
#else
	// wangyu 
	use_workspace = FALSE;
#endif


	/* ---------------------------------------------------------------------- */
	/* compute the norm */
	/* ---------------------------------------------------------------------- */

	xnorm = 0;

	if (use_workspace)
	{
		ERROR(CHOLMOD_INVALID, "not implemented!");
		return (EMPTY);
		/* ------------------------------------------------------------------ */
		/* infinity-norm = max row sum, using stride-1 access of X */
		/* ------------------------------------------------------------------ */

		DEBUG(for (i = 0; i < nrow; i++) ASSERT(W[i] == 0));

		/* this is faster than stride-d, but requires O(nrow) workspace */
		for (j = 0; j < ncol; j++)
		{
			for (i = 0; i < nrow; i++)
			{
				W[i] += abs_value(xtype, Xx, Xz, i + j * d, Common);
			}
		}
		for (i = 0; i < nrow; i++)
		{
			s = W[i];
			if ((IS_NAN(s) || s > xnorm) && !IS_NAN(xnorm))
			{
				xnorm = s;
			}
			W[i] = 0;
		}

	}
	else if (norm == 0)
	{
		ERROR(CHOLMOD_INVALID, "not implemented!");
		return (EMPTY);
		/* ------------------------------------------------------------------ */
		/* infinity-norm = max row sum, using stride-d access of X */
		/* ------------------------------------------------------------------ */

		for (i = 0; i < nrow; i++)
		{
			s = 0;
			for (j = 0; j < ncol; j++)
			{
				s += abs_value(xtype, Xx, Xz, i + j * d, Common);
			}
			if ((IS_NAN(s) || s > xnorm) && !IS_NAN(xnorm))
			{
				xnorm = s;
			}
		}

	}
	else if (norm == 1)
	{
		ERROR(CHOLMOD_INVALID, "not implemented!");
		return (EMPTY);
		/* ------------------------------------------------------------------ */
		/* 1-norm = max column sum */
		/* ------------------------------------------------------------------ */

		for (j = 0; j < ncol; j++)
		{
			s = 0;
			for (i = 0; i < nrow; i++)
			{
				s += abs_value(xtype, Xx, Xz, i + j * d, Common);
			}
			if ((IS_NAN(s) || s > xnorm) && !IS_NAN(xnorm))
			{
				xnorm = s;
			}
		}
	}
	else
	{

		/* ------------------------------------------------------------------ */
		/* 2-norm = sqrt (sum (X.^2)) */
		/* ------------------------------------------------------------------ */

		switch (xtype)
		{

		case CHOLMOD_REAL:
			ASSERT(nrow == X->d);
			for (i = 0; i < nrow; i++)
			{
				for (j = 0; j < ncol; j++) {
					const int p = i + nrow * j;
					x = Xx[p];
					xnorm += x * x;
				}
			}
			break;

		case CHOLMOD_COMPLEX:
			ERROR(CHOLMOD_INVALID, "not implemented!");
			return (EMPTY);
			for (i = 0; i < nrow; i++)
			{
				x = Xx[2 * i];
				z = Xx[2 * i + 1];
				xnorm += x * x + z * z;
			}
			break;

		case CHOLMOD_ZOMPLEX:
			ERROR(CHOLMOD_INVALID, "not implemented!");
			return (EMPTY);
			for (i = 0; i < nrow; i++)
			{
				x = Xx[i];
				z = Xz[i];
				xnorm += x * x + z * z;
			}
			break;
		}

		xnorm = sqrt(xnorm);
	}

	/* ---------------------------------------------------------------------- */
	/* return result */
	/* ---------------------------------------------------------------------- */

	return (xnorm);
}


double my_cholmod_l_norm_dense_ori //getEntryRef
(
	/* ---- input ---- */
	cholmod_dense* X,	/* matrix to compute the norm of */
	int norm,		/* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
	/* --------------- */
	cholmod_common* Common
	)
{
	double xnorm, s, x, z;
	double* Xx, * Xz, * W;
	Int nrow, ncol, d, i, j, use_workspace, xtype;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	RETURN_IF_NULL_COMMON(EMPTY);
	RETURN_IF_NULL(X, EMPTY);
	RETURN_IF_XTYPE_INVALID(X, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, EMPTY);
	Common->status = CHOLMOD_OK;
	ncol = X->ncol;
	if (norm < 0 || norm > 2 || (norm == 2 && ncol > 1))
	{
		ERROR(CHOLMOD_INVALID, "invalid norm");
		return (EMPTY);
	}

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	nrow = X->nrow;
	d = X->d;
	Xx = (double*) X->x; // wangyu:  Xx = X->x;
	Xz = (double*) X->z; // wangyu:  Xz = X->z;
	xtype = X->xtype;

	/* ---------------------------------------------------------------------- */
	/* allocate workspace, if needed */
	/* ---------------------------------------------------------------------- */

	W = NULL;
	
#if 0
	use_workspace = (norm == 0 && ncol > 4);
	if (use_workspace)
	{
		CHOLMOD(allocate_work) (0, 0, nrow, Common);
		W = (double*) Common->Xwork; // wangyu: W = Common->Xwork;
		if (Common->status < CHOLMOD_OK)
		{
			/* oops, no workspace */
			use_workspace = FALSE;
		}
	}
#else
	// wangyu 
	use_workspace = FALSE; 
#endif


	/* ---------------------------------------------------------------------- */
	/* compute the norm */
	/* ---------------------------------------------------------------------- */

	xnorm = 0;

	if (use_workspace)
	{

		/* ------------------------------------------------------------------ */
		/* infinity-norm = max row sum, using stride-1 access of X */
		/* ------------------------------------------------------------------ */

		DEBUG(for (i = 0; i < nrow; i++) ASSERT(W[i] == 0));

		/* this is faster than stride-d, but requires O(nrow) workspace */
		for (j = 0; j < ncol; j++)
		{
			for (i = 0; i < nrow; i++)
			{
				W[i] += abs_value(xtype, Xx, Xz, i + j * d, Common);
			}
		}
		for (i = 0; i < nrow; i++)
		{
			s = W[i];
			if ((IS_NAN(s) || s > xnorm) && !IS_NAN(xnorm))
			{
				xnorm = s;
			}
			W[i] = 0;
		}

	}
	else if (norm == 0)
	{

		/* ------------------------------------------------------------------ */
		/* infinity-norm = max row sum, using stride-d access of X */
		/* ------------------------------------------------------------------ */

		for (i = 0; i < nrow; i++)
		{
			s = 0;
			for (j = 0; j < ncol; j++)
			{
				s += abs_value(xtype, Xx, Xz, i + j * d, Common);
			}
			if ((IS_NAN(s) || s > xnorm) && !IS_NAN(xnorm))
			{
				xnorm = s;
			}
		}

	}
	else if (norm == 1)
	{

		/* ------------------------------------------------------------------ */
		/* 1-norm = max column sum */
		/* ------------------------------------------------------------------ */

		for (j = 0; j < ncol; j++)
		{
			s = 0;
			for (i = 0; i < nrow; i++)
			{
				s += abs_value(xtype, Xx, Xz, i + j * d, Common);
			}
			if ((IS_NAN(s) || s > xnorm) && !IS_NAN(xnorm))
			{
				xnorm = s;
			}
		}
	}
	else
	{

		/* ------------------------------------------------------------------ */
		/* 2-norm = sqrt (sum (X.^2)) */
		/* ------------------------------------------------------------------ */

		switch (xtype)
		{

		case CHOLMOD_REAL:
			for (i = 0; i < nrow; i++)
			{
				x = Xx[i];
				xnorm += x * x;
			}
			break;

		case CHOLMOD_COMPLEX:
			for (i = 0; i < nrow; i++)
			{
				x = Xx[2 * i];
				z = Xx[2 * i + 1];
				xnorm += x * x + z * z;
			}
			break;

		case CHOLMOD_ZOMPLEX:
			for (i = 0; i < nrow; i++)
			{
				x = Xx[i];
				z = Xz[i];
				xnorm += x * x + z * z;
			}
			break;
		}

		xnorm = sqrt(xnorm);
	}

	/* ---------------------------------------------------------------------- */
	/* return result */
	/* ---------------------------------------------------------------------- */

	return (xnorm);
}


/* ========================================================================== */
/* === cholmod_norm_sparse ================================================== */
/* ========================================================================== */

double my_cholmod_l_norm_sparse
(
	/* ---- input ---- */
	cholmod_sparse* A,	/* matrix to compute the norm of */
	int norm,		/* type of norm: 0: inf. norm, 1: 1-norm */
	/* --------------- */
	cholmod_common* Common
	)
{
	double anorm, s;
	double* Ax, * Az, * W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remote this one or it will return.
	RETURN_IF_NULL(A, EMPTY);
	RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;
	if (norm < 0 || norm > 1)
	{
		ERROR(CHOLMOD_INVALID, "invalid norm");
		return (EMPTY);
	}
	if (A->stype && nrow != ncol)
	{
		ERROR(CHOLMOD_INVALID, "matrix invalid");
		return (EMPTY);
	}

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;
	packed = (Int)A->packed;
	xtype = (Int)A->xtype;

	/* ---------------------------------------------------------------------- */
	/* allocate workspace, if needed */
	/* ---------------------------------------------------------------------- */

	W = NULL;
	if (A->stype || norm == 0)
	{
#if 0
		CHOLMOD(allocate_work) (0, 0, nrow, Common);
		W = (double *)Common->Xwork;
		if (Common->status < CHOLMOD_OK)
		{
			/* out of memory */
			return (EMPTY);
		}
		DEBUG(for (i = 0; i < nrow; i++) ASSERT(W[i] == 0));
#else
		RAISE_ERROR("Not implemented!\n");
#endif
	}

	/* ---------------------------------------------------------------------- */
	/* compute the norm */
	/* ---------------------------------------------------------------------- */

	anorm = 0;

	if (A->stype > 0)
	{

		/* ------------------------------------------------------------------ */
		/* A is symmetric with upper triangular part stored */
		/* ------------------------------------------------------------------ */

		/* infinity-norm = 1-norm = max row/col sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			for (; p < pend; p++)
			{
				i = Ai[p];
				s = abs_value(xtype, Ax, Az, p, Common);
				if (i == j)
				{
					W[i] += s;
				}
				else if (i < j)
				{
					W[i] += s;
					W[j] += s;
				}
			}
		}

	}
	else if (A->stype < 0)
	{

		/* ------------------------------------------------------------------ */
		/* A is symmetric with lower triangular part stored */
		/* ------------------------------------------------------------------ */

		/* infinity-norm = 1-norm = max row/col sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			for (; p < pend; p++)
			{
				i = Ai[p];
				s = abs_value(xtype, Ax, Az, p, Common);
				if (i == j)
				{
					W[i] += s;
				}
				else if (i > j)
				{
					W[i] += s;
					W[j] += s;
				}
			}
		}

	}
	else if (norm == 0)
	{

		/* ------------------------------------------------------------------ */
		/* A is unsymmetric, compute the infinity-norm */
		/* ------------------------------------------------------------------ */

		/* infinity-norm = max row sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			for (; p < pend; p++)
			{
				W[Ai[p]] += abs_value(xtype, Ax, Az, p, Common);
			}
		}

	}
	else
	{

		/* ------------------------------------------------------------------ */
		/* A is unsymmetric, compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			if (xtype == CHOLMOD_PATTERN)
			{
				s = pend - p;
			}
			else
			{
				s = 0;
				for (; p < pend; p++)
				{
					s += abs_value(xtype, Ax, Az, p, Common);
				}
			}
			if ((IS_NAN(s) || s > anorm) && !IS_NAN(anorm))
			{
				anorm = s;
			}
		}
	}

	/* ---------------------------------------------------------------------- */
	/* compute the max row sum */
	/* ---------------------------------------------------------------------- */

	if (A->stype || norm == 0)
	{
#if 0
		for (i = 0; i < nrow; i++)
		{
			s = W[i];
			if ((IS_NAN(s) || s > anorm) && !IS_NAN(anorm))
			{
				anorm = s;
			}
			W[i] = 0;
		}
#else
	RAISE_ERROR("Not implemented!\n");
#endif
	}

	/* ---------------------------------------------------------------------- */
	/* return result */
	/* ---------------------------------------------------------------------- */

	return (anorm);
}


// this is modified from cholmod_l_norm_sparse
double my_cholmod_l_pnorm_sparse
(
	/* ---- input ---- */
	cholmod_sparse* A,	/* matrix to compute the norm of */
	int power,		/* type of norm: 0: inf. norm, 1: 1-norm */
	/* --------------- */
	cholmod_common* Common
)
{

	double s;
	double* Ax, * Az; // *W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remote this one or it will return.
	RETURN_IF_NULL(A, EMPTY);
	RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;
	if (power < 0 || power > 2)
	{
		ERROR(CHOLMOD_INVALID, "invalid power");
		return (EMPTY);
	}
	if (A->stype && nrow != ncol)
	{
		ERROR(CHOLMOD_INVALID, "matrix invalid");
		return (EMPTY);
	}

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;
	packed = (Int)A->packed;
	xtype = (Int)A->xtype;

	/* ---------------------------------------------------------------------- */
	/* allocate workspace, if needed */
	/* ---------------------------------------------------------------------- */


	/* ---------------------------------------------------------------------- */
	/* compute the norm */
	/* ---------------------------------------------------------------------- */

	double sum; 

	if (A->stype > 0)
	{

		/* ------------------------------------------------------------------ */
		/* A is symmetric with upper triangular part stored */
		/* ------------------------------------------------------------------ */

		/* infinity-norm = 1-norm = max row/col sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			for (; p < pend; p++)
			{
				i = Ai[p];
				s = abs_value(xtype, Ax, Az, p, Common);
				if (i == j)
				{
					sum += s; // W[i] += s;
				}
				else if (i < j)
				{
					sum += 2 * s;
					// W[i] += s;
					// W[j] += s;
				}
			}
		}

	}
	else if (A->stype < 0)
	{

		/* ------------------------------------------------------------------ */
		/* A is symmetric with lower triangular part stored */
		/* ------------------------------------------------------------------ */

		/* infinity-norm = 1-norm = max row/col sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			for (; p < pend; p++)
			{
				i = Ai[p];
				s = abs_value(xtype, Ax, Az, p, Common);
				if (i == j)
				{
					sum += s;//W[i] += s;
				}
				else if (i > j)
				{
					sum += 2 * s;
					//W[i] += s;
					//W[j] += s;
				}
			}
		}

	}
	else
	{

		/* ------------------------------------------------------------------ */
		/* A is unsymmetric, compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			if (xtype == CHOLMOD_PATTERN)
			{
				s = pend - p;
			}
			else
			{
				s = 0;
				for (; p < pend; p++)
				{
					s += abs_value(xtype, Ax, Az, p, Common);
				}
			}
			sum += s;
		}
	}


	/* ---------------------------------------------------------------------- */
	/* return result */
	/* ---------------------------------------------------------------------- */

	return (sum);

}


/* ========================================================================== */
/* === my_cholmod_ssmult ======================================================= */
/* ========================================================================== */
//
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
//	)
//{
//	double bjt;
//	double* Ax, * Bx, * Cx, * W;
//	Int* Ap, * Anz, * Ai, * Bp, * Bnz, * Bi, * Cp, * Ci, * Flag;
//	cholmod_sparse* C, * A2, * B2, * A3, * B3, * C2;
//	Int apacked, bpacked, j, i, pa, paend, pb, pbend, ncol, mark, cnz, t, p,
//		nrow, anz, bnz, do_swap_and_transpose, n1, n2;
//
//	/* ---------------------------------------------------------------------- */
//	/* check inputs */
//	/* ---------------------------------------------------------------------- */
//
//	RETURN_IF_NULL_COMMON(NULL);
//	RETURN_IF_NULL(A, NULL);
//	RETURN_IF_NULL(B, NULL);
//	values = values &&
//		(A->xtype != CHOLMOD_PATTERN) && (B->xtype != CHOLMOD_PATTERN);
//	RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN,
//		values ? CHOLMOD_REAL : CHOLMOD_ZOMPLEX, NULL);
//	RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN,
//		values ? CHOLMOD_REAL : CHOLMOD_ZOMPLEX, NULL);
//	if (A->ncol != B->nrow)
//	{
//		/* inner dimensions must agree */
//		ERROR(CHOLMOD_INVALID, "A and B inner dimensions must match");
//		return (NULL);
//	}
//	/* A and B must have the same numerical type if values is TRUE (both must
//	 * be CHOLMOD_REAL, this is implicitly checked above) */
//	Common->status = CHOLMOD_OK;
//
//	/* ---------------------------------------------------------------------- */
//	/* allocate workspace */
//	/* ---------------------------------------------------------------------- */
//
//	if (A->nrow <= 1)
//	{
//		/* C will be implicitly sorted, so no need to sort it here */
//		sorted = FALSE;
//	}
//	if (sorted)
//	{
//		n1 = MAX(A->nrow, B->ncol);
//	}
//	else
//	{
//		n1 = A->nrow;
//	}
//	n2 = MAX4(A->ncol, A->nrow, B->nrow, B->ncol);
//	CHOLMOD(allocate_work) (n1, n2, values ? n1 : 0, Common);
//	if (Common->status < CHOLMOD_OK)
//	{
//		/* out of memory */
//		return (NULL);
//	}
//	// ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common)); // removed by wangyu
//
//	/* ---------------------------------------------------------------------- */
//	/* get inputs */
//	/* ---------------------------------------------------------------------- */
//
//	/* convert A to unsymmetric, if necessary */
//	A2 = NULL;
//	B2 = NULL;
//	if (A->stype)
//	{
//		/* workspace: Iwork (max (A->nrow,A->ncol)) */
//		A2 = CHOLMOD(copy) (A, 0, values, Common);
//		if (Common->status < CHOLMOD_OK)
//		{
//			/* out of memory */
//			// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//			return (NULL);
//		}
//		A = A2;
//	}
//
//	/* convert B to unsymmetric, if necessary */
//	if (B->stype)
//	{
//		/* workspace: Iwork (max (B->nrow,B->ncol)) */
//		B2 = CHOLMOD(copy) (B, 0, values, Common);
//		if (Common->status < CHOLMOD_OK)
//		{
//			/* out of memory */
//			CHOLMOD(free_sparse) (&A2, Common);
//			// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//			return (NULL);
//		}
//		B = B2;
//	}
//
//	// ASSERT(CHOLMOD(dump_sparse) (A, "A", Common) >= 0); // removed by wangyu
//	// ASSERT(CHOLMOD(dump_sparse) (B, "B", Common) >= 0); // removed by wangyu
//
//	/* get the A matrix */
//	Ap = (Int*)A->p;
//	Anz = (Int*)A->nz;
//	Ai = (Int*)A->i;
//	Ax = (double*)A->x;
//	apacked = A->packed;
//
//	/* get the B matrix */
//	Bp = (Int*)B->p;
//	Bnz = (Int*)B->nz;
//	Bi = (Int*)B->i;
//	Bx = (double*)B->x;
//	bpacked = B->packed;
//
//	/* get the size of C */
//	nrow = A->nrow;
//	ncol = B->ncol;
//
//	/* get workspace */
//	W = (double*)Common->Xwork;		/* size nrow, unused if values is FALSE */
//	Flag = (Int*)Common->Flag;	/* size nrow, Flag [0..nrow-1] < mark on input*/
//
//	/* ---------------------------------------------------------------------- */
//	/* count the number of entries in the result C */
//	/* ---------------------------------------------------------------------- */
//
//	cnz = 0;
//	for (j = 0; j < ncol; j++)
//	{
//		/* clear the Flag array */
//		/* mark = CHOLMOD(clear_flag) (Common) ; */
//		CHOLMOD_CLEAR_FLAG(Common);
//		mark = Common->mark;
//
//		/* for each nonzero B(t,j) in column j, do: */
//		pb = Bp[j];
//		pbend = (bpacked) ? (Bp[j + 1]) : (pb + Bnz[j]);
//		for (; pb < pbend; pb++)
//		{
//			/* B(t,j) is nonzero */
//			t = Bi[pb];
//
//			/* add the nonzero pattern of A(:,t) to the pattern of C(:,j) */
//			pa = Ap[t];
//			paend = (apacked) ? (Ap[t + 1]) : (pa + Anz[t]);
//			for (; pa < paend; pa++)
//			{
//				i = Ai[pa];
//				if (Flag[i] != mark)
//				{
//					Flag[i] = mark;
//					cnz++;
//				}
//			}
//		}
//		if (cnz < 0)
//		{
//			break;	    /* integer overflow case */
//		}
//	}
//
//	/* mark = CHOLMOD(clear_flag) (Common) ; */
//	CHOLMOD_CLEAR_FLAG(Common);
//	mark = Common->mark;
//
//	/* ---------------------------------------------------------------------- */
//	/* check for integer overflow */
//	/* ---------------------------------------------------------------------- */
//
//	if (cnz < 0)
//	{
//		ERROR(CHOLMOD_TOO_LARGE, "problem too large");
//		CHOLMOD(free_sparse) (&A2, Common);
//		CHOLMOD(free_sparse) (&B2, Common);
//		// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//		return (NULL);
//	}
//
//	/* ---------------------------------------------------------------------- */
//	/* Determine how to return C sorted (if requested) */
//	/* ---------------------------------------------------------------------- */
//
//	do_swap_and_transpose = FALSE;
//
//	if (sorted)
//	{
//		/* Determine the best way to return C with sorted columns.  Computing
//		 * C = (B'*A')' takes cnz + anz + bnz time (ignoring O(n) terms).
//		 * Sorting C when done, C = (A*B)'', takes 2*cnz time.  Pick the one
//		 * with the least amount of work. */
//
//		anz = CHOLMOD(nnz) (A, Common);
//		bnz = CHOLMOD(nnz) (B, Common);
//
//		do_swap_and_transpose = (anz + bnz < cnz);
//
//		if (do_swap_and_transpose)
//		{
//
//			/* -------------------------------------------------------------- */
//			/* C = (B'*A')' */
//			/* -------------------------------------------------------------- */
//
//			/* workspace: Iwork (A->nrow) */
//			A3 = CHOLMOD(ptranspose) (A, values, NULL, NULL, 0, Common);
//			CHOLMOD(free_sparse) (&A2, Common);
//			A2 = A3;
//			if (Common->status < CHOLMOD_OK)
//			{
//				/* out of memory */
//				CHOLMOD(free_sparse) (&A2, Common);
//				CHOLMOD(free_sparse) (&B2, Common);
//				// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//				return (NULL);
//			}
//			/* workspace: Iwork (B->nrow) */
//			B3 = CHOLMOD(ptranspose) (B, values, NULL, NULL, 0, Common);
//			CHOLMOD(free_sparse) (&B2, Common);
//			B2 = B3;
//			if (Common->status < CHOLMOD_OK)
//			{
//				/* out of memory */
//				CHOLMOD(free_sparse) (&A2, Common);
//				CHOLMOD(free_sparse) (&B2, Common);
//				// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//				return (NULL);
//			}
//			A = B2;
//			B = A2;
//
//			/* get the new A matrix */
//			Ap = (Int*)A->p;
//			Anz = (Int*)A->nz;
//			Ai = (Int*)A->i;
//			Ax = (double*)A->x;
//			apacked = A->packed;
//
//			/* get the new B matrix */
//			Bp = (Int*)B->p;
//			Bnz = (Int*)B->nz;
//			Bi = (Int*)B->i;
//			Bx = (double*)B->x;
//			bpacked = B->packed;
//
//			/* get the size of C' */
//			nrow = A->nrow;
//			ncol = B->ncol;
//		}
//	}
//
//	/* ---------------------------------------------------------------------- */
//	/* allocate C */
//	/* ---------------------------------------------------------------------- */
//
//	C = CHOLMOD(allocate_sparse) (nrow, ncol, cnz, FALSE, TRUE, 0,
//		values ? A->xtype : CHOLMOD_PATTERN, Common);
//	if (Common->status < CHOLMOD_OK)
//	{
//		/* out of memory */
//		CHOLMOD(free_sparse) (&A2, Common);
//		CHOLMOD(free_sparse) (&B2, Common);
//		// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//		return (NULL);
//	}
//
//	Cp = (Int*)C->p;
//	Ci = (Int*)C->i;
//	Cx = (double*)C->x;
//
//	/* ---------------------------------------------------------------------- */
//	/* C = A*B */
//	/* ---------------------------------------------------------------------- */
//
//	cnz = 0;
//
//	if (values)
//	{
//
//		/* pattern and values */
//		for (j = 0; j < ncol; j++)
//		{
//			/* clear the Flag array */
//			/* mark = CHOLMOD(clear_flag (Common)) ; */
//			CHOLMOD_CLEAR_FLAG(Common);
//			mark = Common->mark;
//
//			/* start column j of C */
//			Cp[j] = cnz;
//
//			/* for each nonzero B(t,j) in column j, do: */
//			pb = Bp[j];
//			pbend = (bpacked) ? (Bp[j + 1]) : (pb + Bnz[j]);
//			for (; pb < pbend; pb++)
//			{
//				/* B(t,j) is nonzero */
//				t = Bi[pb];
//				bjt = Bx[pb];
//
//				/* add the nonzero pattern of A(:,t) to the pattern of C(:,j)
//				 * and scatter the values into W */
//				pa = Ap[t];
//				paend = (apacked) ? (Ap[t + 1]) : (pa + Anz[t]);
//				for (; pa < paend; pa++)
//				{
//					i = Ai[pa];
//					if (Flag[i] != mark)
//					{
//						Flag[i] = mark;
//						Ci[cnz++] = i;
//					}
//					W[i] += Ax[pa] * bjt;
//				}
//			}
//
//			/* gather the values into C(:,j) */
//			for (p = Cp[j]; p < cnz; p++)
//			{
//				i = Ci[p];
//				Cx[p] = W[i];
//				W[i] = 0;
//			}
//		}
//
//	}
//	else
//	{
//
//		/* pattern only */
//		for (j = 0; j < ncol; j++)
//		{
//			/* clear the Flag array */
//			/* mark = CHOLMOD(clear_flag) (Common) ; */
//			CHOLMOD_CLEAR_FLAG(Common);
//			mark = Common->mark;
//
//			/* start column j of C */
//			Cp[j] = cnz;
//
//			/* for each nonzero B(t,j) in column j, do: */
//			pb = Bp[j];
//			pbend = (bpacked) ? (Bp[j + 1]) : (pb + Bnz[j]);
//			for (; pb < pbend; pb++)
//			{
//				/* B(t,j) is nonzero */
//				t = Bi[pb];
//
//				/* add the nonzero pattern of A(:,t) to the pattern of C(:,j) */
//				pa = Ap[t];
//				paend = (apacked) ? (Ap[t + 1]) : (pa + Anz[t]);
//				for (; pa < paend; pa++)
//				{
//					i = Ai[pa];
//					if (Flag[i] != mark)
//					{
//						Flag[i] = mark;
//						Ci[cnz++] = i;
//					}
//				}
//			}
//		}
//	}
//
//	Cp[ncol] = cnz;
//	ASSERT(MAX(1, cnz) == C->nzmax);
//
//	/* ---------------------------------------------------------------------- */
//	/* clear workspace and free temporary matrices */
//	/* ---------------------------------------------------------------------- */
//
//	CHOLMOD(free_sparse) (&A2, Common);
//	CHOLMOD(free_sparse) (&B2, Common);
//	/* CHOLMOD(clear_flag) (Common) ; */
//	CHOLMOD_CLEAR_FLAG(Common);
//	// wangyu removed:  ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//
//	/* ---------------------------------------------------------------------- */
//	/* convert C to a symmetric upper/lower matrix if requested */
//	/* ---------------------------------------------------------------------- */
//
//	/* convert C in place, which cannot fail since no memory is allocated */
//	if (stype > 0)
//	{
//		/* C = triu (C), in place */
//		(void)CHOLMOD(band_inplace) (0, ncol, values, C, Common);
//		C->stype = 1;
//	}
//	else if (stype < 0)
//	{
//		/* C = tril (C), in place */
//		(void)CHOLMOD(band_inplace) (-nrow, 0, values, C, Common);
//		C->stype = -1;
//	}
//	ASSERT(Common->status >= CHOLMOD_OK);
//
//	/* ---------------------------------------------------------------------- */
//	/* sort C, if requested */
//	/* ---------------------------------------------------------------------- */
//
//	if (sorted)
//	{
//		if (do_swap_and_transpose)
//		{
//			/* workspace: Iwork (C->ncol), which is A->nrow since C=(B'*A') */
//			C2 = CHOLMOD(ptranspose) (C, values, NULL, NULL, 0, Common);
//			CHOLMOD(free_sparse) (&C, Common);
//			if (Common->status < CHOLMOD_OK)
//			{
//				/* out of memory */
//				// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//				return (NULL);
//			}
//			C = C2;
//		}
//		else
//		{
//			/* workspace: Iwork (max (C->nrow,C->ncol)) */
//			if (!CHOLMOD(sort) (C, Common))
//			{
//				/* out of memory */
//				CHOLMOD(free_sparse) (&C, Common);
//				// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//				return (NULL);
//			}
//		}
//	}
//
//	/* ---------------------------------------------------------------------- */
//	/* return result */
//	/* ---------------------------------------------------------------------- */
//
//	// wangyu removed: DEBUG(CHOLMOD(dump_sparse) (C, "ssmult", Common) >= 0);
//	// wangyu removed: ASSERT(CHOLMOD(dump_work) (TRUE, TRUE, values ? n1 : 0, Common));
//	return (C);
//}
//

/* ========================================================================== */
/* === cholmod_sdmult ======================================================= */
/* ========================================================================== */

#include "cholmod_memory.cpp" // wangyu put it here. 

#define REAL
#include "t_mycholmod_sdmult.cpp"
//#define COMPLEX
//#include "t_cholmod_sdmult.c"
//#define ZOMPLEX
//#include "t_cholmod_sdmult.c"

//int CHOLMOD(sdmult)
int my_cholmod_l_sdmult
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
	)
{
	double* w;
	size_t nx, ny;
	Int e;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	RETURN_IF_NULL_COMMON(FALSE);
	RETURN_IF_NULL(A, FALSE);
	RETURN_IF_NULL(X, FALSE);
	RETURN_IF_NULL(Y, FALSE);
	RETURN_IF_XTYPE_INVALID(A, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE);
	RETURN_IF_XTYPE_INVALID(X, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE);
	RETURN_IF_XTYPE_INVALID(Y, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE);
	ny = transpose ? A->ncol : A->nrow;	/* required length of Y */
	nx = transpose ? A->nrow : A->ncol;	/* required length of X */
	if (X->nrow != nx || X->ncol != Y->ncol || Y->nrow != ny)
	{
		/* X and/or Y have the wrong dimension */
		ERROR(CHOLMOD_INVALID, "X and/or Y have wrong dimensions");
		return (FALSE);
	}
	if (A->xtype != X->xtype || A->xtype != Y->xtype)
	{
		ERROR(CHOLMOD_INVALID, "A, X, and Y must have same xtype");
		return (FALSE);
	}
	Common->status = CHOLMOD_OK;

	/* ---------------------------------------------------------------------- */
	/* allocate workspace, if required */
	/* ---------------------------------------------------------------------- */

	w = NULL;
	e = (A->xtype == CHOLMOD_REAL ? 1 : 2);
	if (A->stype && X->ncol >= 4)
	{
		w = (double*) CHOLMOD(malloc) (nx, 4 * e * sizeof(double), Common);
	}
	if (Common->status < CHOLMOD_OK)
	{
		return (FALSE);    /* out of memory */
	}

	/* ---------------------------------------------------------------------- */
	/* Y = alpha*op(A)*X + beta*Y via template routine */
	/* ---------------------------------------------------------------------- */

	//ASSERT(CHOLMOD(dump_sparse) (A, "A", Common) >= 0); // wangyu removed
	DEBUG(CHOLMOD(dump_dense) (X, "X", Common)); 
	DEBUG(if (IS_NONZERO(beta[0])
		|| (IS_NONZERO(beta[1]) && A->xtype != CHOLMOD_REAL))
		CHOLMOD(dump_dense) (Y, "Y", Common));

	ASSERT(A->xtype == CHOLMOD_REAL); // wangyu
	switch (A->xtype)
	{

	case CHOLMOD_REAL:
		r_cholmod_sdmult(A, transpose, alpha, beta, X, Y, w);
		break;
		/* wangyu: remove non-real case. 
	case CHOLMOD_COMPLEX:
		c_cholmod_sdmult(A, transpose, alpha, beta, X, Y, w);
		break;

	case CHOLMOD_ZOMPLEX:
		z_cholmod_sdmult(A, transpose, alpha, beta, X, Y, w);
		break;
		*/
	}

	/* ---------------------------------------------------------------------- */
	/* free workspace */
	/* ---------------------------------------------------------------------- */

	CHOLMOD(free) (4 * nx, e * sizeof(double), w, Common);
	DEBUG(CHOLMOD(dump_dense) (Y, "Y", Common));
	return (TRUE);
}

// only consider real case now
#define ADVANCE(x,z,d) x += d
static void r_cholmod_sdmult_simple
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
	)
{

	double yx[8], xx[8], ax[2];
#ifdef ZOMPLEX
	double yz[4], xz[4], az[1];
	double betaz[1], alphaz[1];
#endif

	double* Ax, * Az, * Xx, * Xz, * Yx, * Yz, * w, * Wz;
	Int* Ap, * Ai, * Anz;
	size_t nx, ny, dx, dy;
	Int packed, nrow, ncol, j, k, p, pend, kcol, i;

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

#ifdef ZOMPLEX
	betaz[0] = beta[1];
	alphaz[0] = alpha[1];
#endif

	ny = transpose ? A->ncol : A->nrow;	/* required length of Y */
	nx = transpose ? A->nrow : A->ncol;	/* required length of X */

	nrow = A->nrow;
	ncol = A->ncol;

	Ap = (Int*)A->p;
	Anz = (Int*)A->nz;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	packed = A->packed;
	Xx = (double*)X->x;
	Xz = (double*)X->z;
	Yx = (double*)Y->x;
	Yz = (double*)Y->z;
	kcol = X->ncol;
	dy = Y->d;
	dx = X->d;
	w = W;
	Wz = W + 4 * nx;

	/* ---------------------------------------------------------------------- */
	/* Y = beta * Y */
	/* ---------------------------------------------------------------------- */

	if (ENTRY_IS_ZERO(beta, betaz, 0))
	{
		for (k = 0; k < kcol; k++)
		{
			for (i = 0; i < ((Int)ny); i++)
			{
				/* y [i] = 0. ; */
				CLEAR(Yx, Yz, i);
			}
			/* y += dy ; */
			ADVANCE(Yx, Yz, dy);
		}
	}
	else if (!ENTRY_IS_ONE(beta, betaz, 0))
	{
		for (k = 0; k < kcol; k++)
		{
			for (i = 0; i < ((Int)ny); i++)
			{
				/* y [i] *= beta [0] ; */
				MULT(Yx, Yz, i, Yx, Yz, i, beta, betaz, 0);
			}
			/* y += dy ; */
			ADVANCE(Yx, Yz, dy);
		}
	}

	if (ENTRY_IS_ZERO(alpha, alphaz, 0))
	{
		/* nothing else to do */
		return;
	}

	/* ---------------------------------------------------------------------- */
	/* Y += alpha * op(A) * X, where op(A)=A or A' */
	/* ---------------------------------------------------------------------- */

	Yx = (double*)Y->x;
	Yz = (double*)Y->z;

	k = 0;

	if (A->stype == 0)
	{

		if (transpose)
		{

			/* -------------------------------------------------------------- */
			/* Y += alpha * A' * x, unsymmetric case */
			/* -------------------------------------------------------------- */

			if (kcol % 4 == 1)
			{

				for (j = 0; j < ncol; j++)
				{
					/* yj = 0. ; */
					CLEAR(yx, yz, 0);
					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						/* yj += conj(Ax [p]) * x [Ai [p]] ; */
						i = Ai[p];
						ASSIGN_CONJ(ax, az, 0, Ax, Az, p);
						MULTADD(yx, yz, 0, ax, az, 0, Xx, Xz, i);
					}
					/* y [j] += alpha [0] * yj ; */
					MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
				}
				/* y += dy ; */
				/* x += dx ; */
				ADVANCE(Yx, Yz, dy);
				ADVANCE(Xx, Xz, dx);
				k++;

			}
			else if (kcol % 4 == 2)
			{

				for (j = 0; j < ncol; j++)
				{
					/* yj0 = 0. ; */
					/* yj1 = 0. ; */
					CLEAR(yx, yz, 0);
					CLEAR(yx, yz, 1);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = conj (Ax [p]) ; */
						ASSIGN_CONJ(ax, az, 0, Ax, Az, p);

						/* yj0 += aij * x [i   ] ; */
						/* yj1 += aij * x [i+dx] ; */
						MULTADD(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADD(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);
					}
					/* y [j   ] += alpha [0] * yj0 ; */
					/* y [j+dy] += alpha [0] * yj1 ; */
					MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
					MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
				}
				/* y += 2*dy ; */
				/* x += 2*dx ; */
				ADVANCE(Yx, Yz, 2 * dy);
				ADVANCE(Xx, Xz, 2 * dx);
				k += 2;

			}
			else if (kcol % 4 == 3)
			{

				for (j = 0; j < ncol; j++)
				{
					/* yj0 = 0. ; */
					/* yj1 = 0. ; */
					/* yj2 = 0. ; */
					CLEAR(yx, yz, 0);
					CLEAR(yx, yz, 1);
					CLEAR(yx, yz, 2);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = conj (Ax [p]) ; */
						ASSIGN_CONJ(ax, az, 0, Ax, Az, p);

						/* yj0 += aij * x [i     ] ; */
						/* yj1 += aij * x [i+  dx] ; */
						/* yj2 += aij * x [i+2*dx] ; */
						MULTADD(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADD(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);
						MULTADD(yx, yz, 2, ax, az, 0, Xx, Xz, i + 2 * dx);
					}
					/* y [j     ] += alpha [0] * yj0 ; */
					/* y [j+  dy] += alpha [0] * yj1 ; */
					/* y [j+2*dy] += alpha [0] * yj2 ; */
					MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
					MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
					MULTADD(Yx, Yz, j + 2 * dy, alpha, alphaz, 0, yx, yz, 2);
				}
				/* y += 3*dy ; */
				/* x += 3*dx ; */
				ADVANCE(Yx, Yz, 3 * dy);
				ADVANCE(Xx, Xz, 3 * dx);
				k += 3;
			}

			for (; k < kcol; k += 4)
			{
				for (j = 0; j < ncol; j++)
				{
					/* yj0 = 0. ; */
					/* yj1 = 0. ; */
					/* yj2 = 0. ; */
					/* yj3 = 0. ; */
					CLEAR(yx, yz, 0);
					CLEAR(yx, yz, 1);
					CLEAR(yx, yz, 2);
					CLEAR(yx, yz, 3);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = conj(Ax [p]) ; */
						ASSIGN_CONJ(ax, az, 0, Ax, Az, p);

						/* yj0 += aij * x [i     ] ; */
						/* yj1 += aij * x [i+  dx] ; */
						/* yj2 += aij * x [i+2*dx] ; */
						/* yj3 += aij * x [i+3*dx] ; */
						MULTADD(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADD(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);
						MULTADD(yx, yz, 2, ax, az, 0, Xx, Xz, i + 2 * dx);
						MULTADD(yx, yz, 3, ax, az, 0, Xx, Xz, i + 3 * dx);

					}
					/* y [j     ] += alpha [0] * yj0 ; */
					/* y [j+  dy] += alpha [0] * yj1 ; */
					/* y [j+2*dy] += alpha [0] * yj2 ; */
					/* y [j+3*dy] += alpha [0] * yj3 ; */
					MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
					MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
					MULTADD(Yx, Yz, j + 2 * dy, alpha, alphaz, 0, yx, yz, 2);
					MULTADD(Yx, Yz, j + 3 * dy, alpha, alphaz, 0, yx, yz, 3);
				}
				/* y += 4*dy ; */
				/* x += 4*dx ; */
				ADVANCE(Yx, Yz, 4 * dy);
				ADVANCE(Xx, Xz, 4 * dx);
			}

		}
		else
		{

			/* -------------------------------------------------------------- */
			/* Y += alpha * A * x, unsymmetric case */
			/* -------------------------------------------------------------- */

			if (kcol % 4 == 1)
			{

				for (j = 0; j < ncol; j++)
				{
					/*  xj = alpha [0] * x [j] ; */
					MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						/* y [Ai [p]] += Ax [p] * xj ; */
						i = Ai[p];
						MULTADD(Yx, Yz, i, Ax, Az, p, xx, xz, 0);
					}
				}
				/* y += dy ; */
				/* x += dx ; */
				ADVANCE(Yx, Yz, dy);
				ADVANCE(Xx, Xz, dx);
				k++;

			}
			else if (kcol % 4 == 2)
			{

				for (j = 0; j < ncol; j++)
				{
					/* xj0 = alpha [0] * x [j   ] ; */
					/* xj1 = alpha [0] * x [j+dx] ; */
					MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
					MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i   ] += aij * xj0 ; */
						/* y [i+dy] += aij * xj1 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
					}
				}
				/* y += 2*dy ; */
				/* x += 2*dx ; */
				ADVANCE(Yx, Yz, 2 * dy);
				ADVANCE(Xx, Xz, 2 * dx);
				k += 2;

			}
			else if (kcol % 4 == 3)
			{

				for (j = 0; j < ncol; j++)
				{
					/* xj0 = alpha [0] * x [j     ] ; */
					/* xj1 = alpha [0] * x [j+  dx] ; */
					/* xj2 = alpha [0] * x [j+2*dx] ; */
					MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
					MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);
					MULT(xx, xz, 2, alpha, alphaz, 0, Xx, Xz, j + 2 * dx);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
					}
				}
				/* y += 3*dy ; */
				/* x += 3*dx ; */
				ADVANCE(Yx, Yz, 3 * dy);
				ADVANCE(Xx, Xz, 3 * dx);
				k += 3;
			}

			for (; k < kcol; k += 4)
			{
				for (j = 0; j < ncol; j++)
				{
					/* xj0 = alpha [0] * x [j     ] ; */
					/* xj1 = alpha [0] * x [j+  dx] ; */
					/* xj2 = alpha [0] * x [j+2*dx] ; */
					/* xj3 = alpha [0] * x [j+3*dx] ; */
					MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
					MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);
					MULT(xx, xz, 2, alpha, alphaz, 0, Xx, Xz, j + 2 * dx);
					MULT(xx, xz, 3, alpha, alphaz, 0, Xx, Xz, j + 3 * dx);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						/* y [i+3*dy] += aij * xj3 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
						MULTADD(Yx, Yz, i + 3 * dy, ax, az, 0, xx, xz, 3);
					}
				}
				/* y += 4*dy ; */
				/* x += 4*dx ; */
				ADVANCE(Yx, Yz, 4 * dy);
				ADVANCE(Xx, Xz, 4 * dx);
			}
		}

	}
	else
	{

		/* ------------------------------------------------------------------ */
		/* Y += alpha * (A or A') * x, symmetric case (upper/lower) */
		/* ------------------------------------------------------------------ */

		/* Only the upper/lower triangular part and the diagonal of A is used.
		 * Since both x and y are written to in the innermost loop, this
		 * code can experience cache bank conflicts if x is used directly.
		 * Thus, a copy is made of x, four columns at a time, if x has
		 * four or more columns.
		 */

		if (kcol % 4 == 1)
		{

			for (j = 0; j < ncol; j++)
			{
				/* yj = 0. ; */
				CLEAR(yx, yz, 0);

				/* xj = alpha [0] * x [j] ; */
				MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);

				p = Ap[j];
				pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
				for (; p < pend; p++)
				{
					i = Ai[p];
					if (i == j)
					{
						/* y [i] += Ax [p] * xj ; */
						MULTADD(Yx, Yz, i, Ax, Az, p, xx, xz, 0);
					}
					else if ((A->stype > 0 && i < j) || (A->stype < 0 && i > j))
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i] += aij * xj ; */
						/* yj    += aij * x [i] ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADDCONJ(yx, yz, 0, ax, az, 0, Xx, Xz, i);


					}
				}
				/* y [j] += alpha [0] * yj ; */
				MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);

			}
			/* y += dy ; */
			/* x += dx ; */
			ADVANCE(Yx, Yz, dy);
			ADVANCE(Xx, Xz, dx);
			k++;

		}
		else if (kcol % 4 == 2)
		{

			for (j = 0; j < ncol; j++)
			{
				/* yj0 = 0. ; */
				/* yj1 = 0. ; */
				CLEAR(yx, yz, 0);
				CLEAR(yx, yz, 1);

				/* xj0 = alpha [0] * x [j   ] ; */
				/* xj1 = alpha [0] * x [j+dx] ; */
				MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
				MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);

				p = Ap[j];
				pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
				for (; p < pend; p++)
				{
					i = Ai[p];
					if (i == j)
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i   ] += aij * xj0 ; */
						/* y [i+dy] += aij * xj1 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);

					}
					else if ((A->stype > 0 && i < j) || (A->stype < 0 && i > j))
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i   ] += aij * xj0 ; */
						/* y [i+dy] += aij * xj1 ; */
						/* yj0 += aij * x [i   ] ; */
						/* yj1 += aij * x [i+dx] ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADDCONJ(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADDCONJ(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);

					}
				}
				/* y [j   ] += alpha [0] * yj0 ; */
				/* y [j+dy] += alpha [0] * yj1 ; */
				MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
				MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);

			}
			/* y += 2*dy ; */
			/* x += 2*dx ; */
			ADVANCE(Yx, Yz, 2 * dy);
			ADVANCE(Xx, Xz, 2 * dx);
			k += 2;

		}
		else if (kcol % 4 == 3)
		{

			for (j = 0; j < ncol; j++)
			{
				/* yj0 = 0. ; */
				/* yj1 = 0. ; */
				/* yj2 = 0. ; */
				CLEAR(yx, yz, 0);
				CLEAR(yx, yz, 1);
				CLEAR(yx, yz, 2);

				/* xj0 = alpha [0] * x [j     ] ; */
				/* xj1 = alpha [0] * x [j+  dx] ; */
				/* xj2 = alpha [0] * x [j+2*dx] ; */
				MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
				MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);
				MULT(xx, xz, 2, alpha, alphaz, 0, Xx, Xz, j + 2 * dx);

				p = Ap[j];
				pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
				for (; p < pend; p++)
				{
					i = Ai[p];
					if (i == j)
					{

						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);

					}
					else if ((A->stype > 0 && i < j) || (A->stype < 0 && i > j))
					{

						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						/* yj0 += aij * x [i     ] ; */
						/* yj1 += aij * x [i+  dx] ; */
						/* yj2 += aij * x [i+2*dx] ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
						MULTADDCONJ(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADDCONJ(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);
						MULTADDCONJ(yx, yz, 2, ax, az, 0, Xx, Xz, i + 2 * dx);

					}
				}
				/* y [j     ] += alpha [0] * yj0 ; */
				/* y [j+  dy] += alpha [0] * yj1 ; */
				/* y [j+2*dy] += alpha [0] * yj2 ; */
				MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
				MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
				MULTADD(Yx, Yz, j + 2 * dy, alpha, alphaz, 0, yx, yz, 2);

			}
			/* y += 3*dy ; */
			/* x += 3*dx ; */
			ADVANCE(Yx, Yz, 3 * dy);
			ADVANCE(Xx, Xz, 3 * dx);

			k += 3;
		}

		/* copy four columns of X into W, and put in row form */
		for (; k < kcol; k += 4)
		{

			for (j = 0; j < ncol; j++)
			{
				/* w [4*j  ] = x [j     ] ; */
				/* w [4*j+1] = x [j+  dx] ; */
				/* w [4*j+2] = x [j+2*dx] ; */
				/* w [4*j+3] = x [j+3*dx] ; */
				ASSIGN(w, Wz, 4 * j, Xx, Xz, j);
				ASSIGN(w, Wz, 4 * j + 1, Xx, Xz, j + dx);
				ASSIGN(w, Wz, 4 * j + 2, Xx, Xz, j + 2 * dx);
				ASSIGN(w, Wz, 4 * j + 3, Xx, Xz, j + 3 * dx);
			}

			for (j = 0; j < ncol; j++)
			{
				/* yj0 = 0. ; */
				/* yj1 = 0. ; */
				/* yj2 = 0. ; */
				/* yj3 = 0. ; */
				CLEAR(yx, yz, 0);
				CLEAR(yx, yz, 1);
				CLEAR(yx, yz, 2);
				CLEAR(yx, yz, 3);

				/* xj0 = alpha [0] * w [4*j  ] ; */
				/* xj1 = alpha [0] * w [4*j+1] ; */
				/* xj2 = alpha [0] * w [4*j+2] ; */
				/* xj3 = alpha [0] * w [4*j+3] ; */
				MULT(xx, xz, 0, alpha, alphaz, 0, w, Wz, 4 * j);
				MULT(xx, xz, 1, alpha, alphaz, 0, w, Wz, 4 * j + 1);
				MULT(xx, xz, 2, alpha, alphaz, 0, w, Wz, 4 * j + 2);
				MULT(xx, xz, 3, alpha, alphaz, 0, w, Wz, 4 * j + 3);

				p = Ap[j];
				pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
				for (; p < pend; p++)
				{
					i = Ai[p];
					if (i == j)
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						/* y [i+3*dy] += aij * xj3 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
						MULTADD(Yx, Yz, i + 3 * dy, ax, az, 0, xx, xz, 3);

					}
					else if ((A->stype > 0 && i < j) || (A->stype < 0 && i > j))
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						/* y [i+3*dy] += aij * xj3 ; */
						/* yj0 += aij * w [4*i  ] ; */
						/* yj1 += aij * w [4*i+1] ; */
						/* yj2 += aij * w [4*i+2] ; */
						/* yj3 += aij * w [4*i+3] ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
						MULTADD(Yx, Yz, i + 3 * dy, ax, az, 0, xx, xz, 3);
						MULTADDCONJ(yx, yz, 0, ax, az, 0, w, Wz, 4 * i);
						MULTADDCONJ(yx, yz, 1, ax, az, 0, w, Wz, 4 * i + 1);
						MULTADDCONJ(yx, yz, 2, ax, az, 0, w, Wz, 4 * i + 2);
						MULTADDCONJ(yx, yz, 3, ax, az, 0, w, Wz, 4 * i + 3);

					}
				}
				/* y [j     ] += alpha [0] * yj0 ; */
				/* y [j+  dy] += alpha [0] * yj1 ; */
				/* y [j+2*dy] += alpha [0] * yj2 ; */
				/* y [j+3*dy] += alpha [0] * yj3 ; */
				MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
				MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
				MULTADD(Yx, Yz, j + 2 * dy, alpha, alphaz, 0, yx, yz, 2);
				MULTADD(Yx, Yz, j + 3 * dy, alpha, alphaz, 0, yx, yz, 3);

			}
			/* y += 4*dy ; */
			/* x += 4*dx ; */
			ADVANCE(Yx, Yz, 4 * dy);
			ADVANCE(Xx, Xz, 4 * dx);

		}
	}
}
#undef ADVANCE


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
)
{
	double* w;
	size_t nx, ny;
	Int e;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	RETURN_IF_NULL_COMMON(FALSE);
	RETURN_IF_NULL(A, FALSE);
	RETURN_IF_NULL(X, FALSE);
	RETURN_IF_NULL(Y, FALSE);
	RETURN_IF_XTYPE_INVALID(A, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE);
	RETURN_IF_XTYPE_INVALID(X, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE);
	RETURN_IF_XTYPE_INVALID(Y, CHOLMOD_REAL, CHOLMOD_ZOMPLEX, FALSE);
	ny = transpose ? A->ncol : A->nrow;	/* required length of Y */
	nx = transpose ? A->nrow : A->ncol;	/* required length of X */
	if (X->nrow != nx || X->ncol != Y->ncol || Y->nrow != ny)
	{
		/* X and/or Y have the wrong dimension */
		ERROR(CHOLMOD_INVALID, "X and/or Y have wrong dimensions");
		return (FALSE);
	}
	if (A->xtype != X->xtype || A->xtype != Y->xtype)
	{
		ERROR(CHOLMOD_INVALID, "A, X, and Y must have same xtype");
		return (FALSE);
	}
	Common->status = CHOLMOD_OK;

	/* ---------------------------------------------------------------------- */
	/* allocate workspace, if required */
	/* ---------------------------------------------------------------------- */

	w = NULL;
	e = (A->xtype == CHOLMOD_REAL ? 1 : 2);
	if (A->stype && X->ncol >= 4)
	{
		w = (double*)CHOLMOD(malloc) (nx, 4 * e * sizeof(double), Common);
	}
	if (Common->status < CHOLMOD_OK)
	{
		return (FALSE);    /* out of memory */
	}

	/* ---------------------------------------------------------------------- */
	/* Y = alpha*op(A)*X + beta*Y via template routine */
	/* ---------------------------------------------------------------------- */

	// ASSERT(CHOLMOD(dump_sparse) (A, "A", Common) >= 0); // wangyu removed
	DEBUG(CHOLMOD(dump_dense) (X, "X", Common));
	DEBUG(if (IS_NONZERO(beta[0])
		|| (IS_NONZERO(beta[1]) && A->xtype != CHOLMOD_REAL))
		CHOLMOD(dump_dense) (Y, "Y", Common));

	ASSERT(A->xtype == CHOLMOD_REAL); // wangyu

	// r_cholmod_sdmult_simple(A, transpose, alpha, beta, X, Y, w);
	r_cholmod_sdmult_mkl(A, transpose, alpha, beta, X, Y, w);

	/* ---------------------------------------------------------------------- */
	/* free workspace */
	/* ---------------------------------------------------------------------- */

	CHOLMOD(free) (4 * nx, e * sizeof(double), w, Common);
	DEBUG(CHOLMOD(dump_dense) (Y, "Y", Common));
	return (TRUE);
}

#include <omp.h>
// only consider real case now
#define ADVANCE(x,z,d) x += d
static void r_cholmod_sdmult_omp
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
)
{

	double yx[8], xx[8], ax[2];
#ifdef ZOMPLEX
	double yz[4], xz[4], az[1];
	double betaz[1], alphaz[1];
#endif

	double* Ax, * Az, * Xx, * Xz, * Yx, * Yz, * w, * Wz;
	Int* Ap, * Ai, * Anz;
	size_t nx, ny, dx, dy;
	Int packed, nrow, ncol, j, k, p, pend, kcol, i;

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

#ifdef ZOMPLEX
	betaz[0] = beta[1];
	alphaz[0] = alpha[1];
#endif

	ny = transpose ? A->ncol : A->nrow;	/* required length of Y */
	nx = transpose ? A->nrow : A->ncol;	/* required length of X */

	nrow = A->nrow;
	ncol = A->ncol;

	Ap = (Int*)A->p;
	Anz = (Int*)A->nz;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	packed = A->packed;
	Xx = (double*)X->x;
	Xz = (double*)X->z;
	Yx = (double*)Y->x;
	Yz = (double*)Y->z;
	kcol = X->ncol;
	dy = Y->d;
	dx = X->d;
	w = W;
	Wz = W + 4 * nx;

	/* ---------------------------------------------------------------------- */
	/* Y = beta * Y */
	/* ---------------------------------------------------------------------- */

	if (ENTRY_IS_ZERO(beta, betaz, 0))
	{
		for (k = 0; k < kcol; k++)
		{
			for (i = 0; i < ((Int)ny); i++)
			{
				/* y [i] = 0. ; */
				CLEAR(Yx, Yz, i);
			}
			/* y += dy ; */
			ADVANCE(Yx, Yz, dy);
		}
	}
	else if (!ENTRY_IS_ONE(beta, betaz, 0))
	{
		for (k = 0; k < kcol; k++)
		{
			for (i = 0; i < ((Int)ny); i++)
			{
				/* y [i] *= beta [0] ; */
				MULT(Yx, Yz, i, Yx, Yz, i, beta, betaz, 0);
			}
			/* y += dy ; */
			ADVANCE(Yx, Yz, dy);
		}
	}

	if (ENTRY_IS_ZERO(alpha, alphaz, 0))
	{
		/* nothing else to do */
		return;
	}

	/* ---------------------------------------------------------------------- */
	/* Y += alpha * op(A) * X, where op(A)=A or A' */
	/* ---------------------------------------------------------------------- */

	Yx = (double*)Y->x;
	Yz = (double*)Y->z;

	k = 0;

	if (A->stype == 0)
	{

		if (transpose)
		{

			/* -------------------------------------------------------------- */
			/* Y += alpha * A' * x, unsymmetric case */
			/* -------------------------------------------------------------- */

			if (kcol % 4 == 1)
			{

				for (j = 0; j < ncol; j++)
				{
					/* yj = 0. ; */
					CLEAR(yx, yz, 0);
					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						/* yj += conj(Ax [p]) * x [Ai [p]] ; */
						i = Ai[p];
						ASSIGN_CONJ(ax, az, 0, Ax, Az, p);
						MULTADD(yx, yz, 0, ax, az, 0, Xx, Xz, i);
					}
					/* y [j] += alpha [0] * yj ; */
					MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
				}
				/* y += dy ; */
				/* x += dx ; */
				ADVANCE(Yx, Yz, dy);
				ADVANCE(Xx, Xz, dx);
				k++;

			}
			else if (kcol % 4 == 2)
			{

				for (j = 0; j < ncol; j++)
				{
					/* yj0 = 0. ; */
					/* yj1 = 0. ; */
					CLEAR(yx, yz, 0);
					CLEAR(yx, yz, 1);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = conj (Ax [p]) ; */
						ASSIGN_CONJ(ax, az, 0, Ax, Az, p);

						/* yj0 += aij * x [i   ] ; */
						/* yj1 += aij * x [i+dx] ; */
						MULTADD(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADD(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);
					}
					/* y [j   ] += alpha [0] * yj0 ; */
					/* y [j+dy] += alpha [0] * yj1 ; */
					MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
					MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
				}
				/* y += 2*dy ; */
				/* x += 2*dx ; */
				ADVANCE(Yx, Yz, 2 * dy);
				ADVANCE(Xx, Xz, 2 * dx);
				k += 2;

			}
			else if (kcol % 4 == 3)
			{

				for (j = 0; j < ncol; j++)
				{
					/* yj0 = 0. ; */
					/* yj1 = 0. ; */
					/* yj2 = 0. ; */
					CLEAR(yx, yz, 0);
					CLEAR(yx, yz, 1);
					CLEAR(yx, yz, 2);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = conj (Ax [p]) ; */
						ASSIGN_CONJ(ax, az, 0, Ax, Az, p);

						/* yj0 += aij * x [i     ] ; */
						/* yj1 += aij * x [i+  dx] ; */
						/* yj2 += aij * x [i+2*dx] ; */
						MULTADD(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADD(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);
						MULTADD(yx, yz, 2, ax, az, 0, Xx, Xz, i + 2 * dx);
					}
					/* y [j     ] += alpha [0] * yj0 ; */
					/* y [j+  dy] += alpha [0] * yj1 ; */
					/* y [j+2*dy] += alpha [0] * yj2 ; */
					MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
					MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
					MULTADD(Yx, Yz, j + 2 * dy, alpha, alphaz, 0, yx, yz, 2);
				}
				/* y += 3*dy ; */
				/* x += 3*dx ; */
				ADVANCE(Yx, Yz, 3 * dy);
				ADVANCE(Xx, Xz, 3 * dx);
				k += 3;
			}

			for (; k < kcol; k += 4)
			{
				for (j = 0; j < ncol; j++)
				{
					/* yj0 = 0. ; */
					/* yj1 = 0. ; */
					/* yj2 = 0. ; */
					/* yj3 = 0. ; */
					CLEAR(yx, yz, 0);
					CLEAR(yx, yz, 1);
					CLEAR(yx, yz, 2);
					CLEAR(yx, yz, 3);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = conj(Ax [p]) ; */
						ASSIGN_CONJ(ax, az, 0, Ax, Az, p);

						/* yj0 += aij * x [i     ] ; */
						/* yj1 += aij * x [i+  dx] ; */
						/* yj2 += aij * x [i+2*dx] ; */
						/* yj3 += aij * x [i+3*dx] ; */
						MULTADD(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADD(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);
						MULTADD(yx, yz, 2, ax, az, 0, Xx, Xz, i + 2 * dx);
						MULTADD(yx, yz, 3, ax, az, 0, Xx, Xz, i + 3 * dx);

					}
					/* y [j     ] += alpha [0] * yj0 ; */
					/* y [j+  dy] += alpha [0] * yj1 ; */
					/* y [j+2*dy] += alpha [0] * yj2 ; */
					/* y [j+3*dy] += alpha [0] * yj3 ; */
					MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
					MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
					MULTADD(Yx, Yz, j + 2 * dy, alpha, alphaz, 0, yx, yz, 2);
					MULTADD(Yx, Yz, j + 3 * dy, alpha, alphaz, 0, yx, yz, 3);
				}
				/* y += 4*dy ; */
				/* x += 4*dx ; */
				ADVANCE(Yx, Yz, 4 * dy);
				ADVANCE(Xx, Xz, 4 * dx);
			}

		}
		else
		{

			/* -------------------------------------------------------------- */
			/* Y += alpha * A * x, unsymmetric case */
			/* -------------------------------------------------------------- */

			if (kcol % 4 == 1)
			{

				for (j = 0; j < ncol; j++)
				{
					/*  xj = alpha [0] * x [j] ; */
					MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						/* y [Ai [p]] += Ax [p] * xj ; */
						i = Ai[p];
						MULTADD(Yx, Yz, i, Ax, Az, p, xx, xz, 0);
					}
				}
				/* y += dy ; */
				/* x += dx ; */
				ADVANCE(Yx, Yz, dy);
				ADVANCE(Xx, Xz, dx);
				k++;

			}
			else if (kcol % 4 == 2)
			{

				for (j = 0; j < ncol; j++)
				{
					/* xj0 = alpha [0] * x [j   ] ; */
					/* xj1 = alpha [0] * x [j+dx] ; */
					MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
					MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i   ] += aij * xj0 ; */
						/* y [i+dy] += aij * xj1 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
					}
				}
				/* y += 2*dy ; */
				/* x += 2*dx ; */
				ADVANCE(Yx, Yz, 2 * dy);
				ADVANCE(Xx, Xz, 2 * dx);
				k += 2;

			}
			else if (kcol % 4 == 3)
			{

				for (j = 0; j < ncol; j++)
				{
					/* xj0 = alpha [0] * x [j     ] ; */
					/* xj1 = alpha [0] * x [j+  dx] ; */
					/* xj2 = alpha [0] * x [j+2*dx] ; */
					MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
					MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);
					MULT(xx, xz, 2, alpha, alphaz, 0, Xx, Xz, j + 2 * dx);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
					}
				}
				/* y += 3*dy ; */
				/* x += 3*dx ; */
				ADVANCE(Yx, Yz, 3 * dy);
				ADVANCE(Xx, Xz, 3 * dx);
				k += 3;
			}

			for (; k < kcol; k += 4)
			{
				for (j = 0; j < ncol; j++)
				{
					/* xj0 = alpha [0] * x [j     ] ; */
					/* xj1 = alpha [0] * x [j+  dx] ; */
					/* xj2 = alpha [0] * x [j+2*dx] ; */
					/* xj3 = alpha [0] * x [j+3*dx] ; */
					MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
					MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);
					MULT(xx, xz, 2, alpha, alphaz, 0, Xx, Xz, j + 2 * dx);
					MULT(xx, xz, 3, alpha, alphaz, 0, Xx, Xz, j + 3 * dx);

					p = Ap[j];
					pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
					for (; p < pend; p++)
					{
						i = Ai[p];
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						/* y [i+3*dy] += aij * xj3 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
						MULTADD(Yx, Yz, i + 3 * dy, ax, az, 0, xx, xz, 3);
					}
				}
				/* y += 4*dy ; */
				/* x += 4*dx ; */
				ADVANCE(Yx, Yz, 4 * dy);
				ADVANCE(Xx, Xz, 4 * dx);
			}
		}

	}
	else
	{

		/* ------------------------------------------------------------------ */
		/* Y += alpha * (A or A') * x, symmetric case (upper/lower) */
		/* ------------------------------------------------------------------ */

		/* Only the upper/lower triangular part and the diagonal of A is used.
		 * Since both x and y are written to in the innermost loop, this
		 * code can experience cache bank conflicts if x is used directly.
		 * Thus, a copy is made of x, four columns at a time, if x has
		 * four or more columns.
		 */

		if (kcol % 4 == 1)
		{

			for (j = 0; j < ncol; j++)
			{
				/* yj = 0. ; */
				CLEAR(yx, yz, 0);

				/* xj = alpha [0] * x [j] ; */
				MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);

				p = Ap[j];
				pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
				for (; p < pend; p++)
				{
					i = Ai[p];
					if (i == j)
					{
						/* y [i] += Ax [p] * xj ; */
						MULTADD(Yx, Yz, i, Ax, Az, p, xx, xz, 0);
					}
					else if ((A->stype > 0 && i < j) || (A->stype < 0 && i > j))
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i] += aij * xj ; */
						/* yj    += aij * x [i] ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADDCONJ(yx, yz, 0, ax, az, 0, Xx, Xz, i);


					}
				}
				/* y [j] += alpha [0] * yj ; */
				MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);

			}
			/* y += dy ; */
			/* x += dx ; */
			ADVANCE(Yx, Yz, dy);
			ADVANCE(Xx, Xz, dx);
			k++;

		}
		else if (kcol % 4 == 2)
		{

			for (j = 0; j < ncol; j++)
			{
				/* yj0 = 0. ; */
				/* yj1 = 0. ; */
				CLEAR(yx, yz, 0);
				CLEAR(yx, yz, 1);

				/* xj0 = alpha [0] * x [j   ] ; */
				/* xj1 = alpha [0] * x [j+dx] ; */
				MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
				MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);

				p = Ap[j];
				pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
				for (; p < pend; p++)
				{
					i = Ai[p];
					if (i == j)
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i   ] += aij * xj0 ; */
						/* y [i+dy] += aij * xj1 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);

					}
					else if ((A->stype > 0 && i < j) || (A->stype < 0 && i > j))
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i   ] += aij * xj0 ; */
						/* y [i+dy] += aij * xj1 ; */
						/* yj0 += aij * x [i   ] ; */
						/* yj1 += aij * x [i+dx] ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADDCONJ(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADDCONJ(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);

					}
				}
				/* y [j   ] += alpha [0] * yj0 ; */
				/* y [j+dy] += alpha [0] * yj1 ; */
				MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
				MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);

			}
			/* y += 2*dy ; */
			/* x += 2*dx ; */
			ADVANCE(Yx, Yz, 2 * dy);
			ADVANCE(Xx, Xz, 2 * dx);
			k += 2;

		}
		else if (kcol % 4 == 3)
		{

			for (j = 0; j < ncol; j++)
			{
				/* yj0 = 0. ; */
				/* yj1 = 0. ; */
				/* yj2 = 0. ; */
				CLEAR(yx, yz, 0);
				CLEAR(yx, yz, 1);
				CLEAR(yx, yz, 2);

				/* xj0 = alpha [0] * x [j     ] ; */
				/* xj1 = alpha [0] * x [j+  dx] ; */
				/* xj2 = alpha [0] * x [j+2*dx] ; */
				MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
				MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);
				MULT(xx, xz, 2, alpha, alphaz, 0, Xx, Xz, j + 2 * dx);

				p = Ap[j];
				pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
				for (; p < pend; p++)
				{
					i = Ai[p];
					if (i == j)
					{

						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);

					}
					else if ((A->stype > 0 && i < j) || (A->stype < 0 && i > j))
					{

						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						/* yj0 += aij * x [i     ] ; */
						/* yj1 += aij * x [i+  dx] ; */
						/* yj2 += aij * x [i+2*dx] ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
						MULTADDCONJ(yx, yz, 0, ax, az, 0, Xx, Xz, i);
						MULTADDCONJ(yx, yz, 1, ax, az, 0, Xx, Xz, i + dx);
						MULTADDCONJ(yx, yz, 2, ax, az, 0, Xx, Xz, i + 2 * dx);

					}
				}
				/* y [j     ] += alpha [0] * yj0 ; */
				/* y [j+  dy] += alpha [0] * yj1 ; */
				/* y [j+2*dy] += alpha [0] * yj2 ; */
				MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
				MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
				MULTADD(Yx, Yz, j + 2 * dy, alpha, alphaz, 0, yx, yz, 2);

			}
			/* y += 3*dy ; */
			/* x += 3*dx ; */
			ADVANCE(Yx, Yz, 3 * dy);
			ADVANCE(Xx, Xz, 3 * dx);

			k += 3;
		}

		/* copy four columns of X into W, and put in row form */
		for (; k < kcol; k += 4)
		{

			for (j = 0; j < ncol; j++)
			{
				/* w [4*j  ] = x [j     ] ; */
				/* w [4*j+1] = x [j+  dx] ; */
				/* w [4*j+2] = x [j+2*dx] ; */
				/* w [4*j+3] = x [j+3*dx] ; */
				ASSIGN(w, Wz, 4 * j, Xx, Xz, j);
				ASSIGN(w, Wz, 4 * j + 1, Xx, Xz, j + dx);
				ASSIGN(w, Wz, 4 * j + 2, Xx, Xz, j + 2 * dx);
				ASSIGN(w, Wz, 4 * j + 3, Xx, Xz, j + 3 * dx);
			}

			for (j = 0; j < ncol; j++)
			{
				/* yj0 = 0. ; */
				/* yj1 = 0. ; */
				/* yj2 = 0. ; */
				/* yj3 = 0. ; */
				CLEAR(yx, yz, 0);
				CLEAR(yx, yz, 1);
				CLEAR(yx, yz, 2);
				CLEAR(yx, yz, 3);

				/* xj0 = alpha [0] * w [4*j  ] ; */
				/* xj1 = alpha [0] * w [4*j+1] ; */
				/* xj2 = alpha [0] * w [4*j+2] ; */
				/* xj3 = alpha [0] * w [4*j+3] ; */
				MULT(xx, xz, 0, alpha, alphaz, 0, w, Wz, 4 * j);
				MULT(xx, xz, 1, alpha, alphaz, 0, w, Wz, 4 * j + 1);
				MULT(xx, xz, 2, alpha, alphaz, 0, w, Wz, 4 * j + 2);
				MULT(xx, xz, 3, alpha, alphaz, 0, w, Wz, 4 * j + 3);

				p = Ap[j];
				pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
				for (; p < pend; p++)
				{
					i = Ai[p];
					if (i == j)
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						/* y [i+3*dy] += aij * xj3 ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
						MULTADD(Yx, Yz, i + 3 * dy, ax, az, 0, xx, xz, 3);

					}
					else if ((A->stype > 0 && i < j) || (A->stype < 0 && i > j))
					{
						/* aij = Ax [p] ; */
						ASSIGN(ax, az, 0, Ax, Az, p);

						/* y [i     ] += aij * xj0 ; */
						/* y [i+  dy] += aij * xj1 ; */
						/* y [i+2*dy] += aij * xj2 ; */
						/* y [i+3*dy] += aij * xj3 ; */
						/* yj0 += aij * w [4*i  ] ; */
						/* yj1 += aij * w [4*i+1] ; */
						/* yj2 += aij * w [4*i+2] ; */
						/* yj3 += aij * w [4*i+3] ; */
						MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
						MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
						MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
						MULTADD(Yx, Yz, i + 3 * dy, ax, az, 0, xx, xz, 3);
						MULTADDCONJ(yx, yz, 0, ax, az, 0, w, Wz, 4 * i);
						MULTADDCONJ(yx, yz, 1, ax, az, 0, w, Wz, 4 * i + 1);
						MULTADDCONJ(yx, yz, 2, ax, az, 0, w, Wz, 4 * i + 2);
						MULTADDCONJ(yx, yz, 3, ax, az, 0, w, Wz, 4 * i + 3);

					}
				}
				/* y [j     ] += alpha [0] * yj0 ; */
				/* y [j+  dy] += alpha [0] * yj1 ; */
				/* y [j+2*dy] += alpha [0] * yj2 ; */
				/* y [j+3*dy] += alpha [0] * yj3 ; */
				MULTADD(Yx, Yz, j, alpha, alphaz, 0, yx, yz, 0);
				MULTADD(Yx, Yz, j + dy, alpha, alphaz, 0, yx, yz, 1);
				MULTADD(Yx, Yz, j + 2 * dy, alpha, alphaz, 0, yx, yz, 2);
				MULTADD(Yx, Yz, j + 3 * dy, alpha, alphaz, 0, yx, yz, 3);

			}
			/* y += 4*dy ; */
			/* x += 4*dx ; */
			ADVANCE(Yx, Yz, 4 * dy);
			ADVANCE(Xx, Xz, 4 * dx);

		}
	}
}
#undef ADVANCE

/* ========================================================================== */
/* === starting from here are functions without alternatives in cholmod ================================================== */
/* ========================================================================== */

void my_cholmod_l_sparse_tensor_diag_mul2
(
cholmod_sparse* A,
double* h, // length of array should be dim * f
cholmod_sparse* B,
int option
)
{
	
	double s;
	double* Ax, * Az; // *W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	double* Bx, * Bz;
	Int* Bp, * Bi, * Bnz;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;

	ASSERT(A->ncol == B->ncol);
	ASSERT(A->nrow == B->nrow);

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;

	Bp = (Int*)B->p;
	Bi = (Int*)B->i;
	Bx = (double*)B->x;
	Bz = (double*)B->z;
	Bnz = (Int*)B->nz;

	packed = (Int)A->packed;
	xtype = (Int)A->xtype;


	ASSERT(A->nz == B->nz);
	ASSERT(A->packed == B->packed);
	ASSERT(A->xtype == B->xtype);

	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;

	//Int f = ncol / dim;
	//ASSERT(ncol == dim * f);
	//ASSERT(dim == 2);

	ASSERT(xtype == CHOLMOD_REAL);
	ASSERT(A->stype == 0); // A is unsymmetric
	{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			ASSERT(Ap[j] == Bp[j]);
			ASSERT(Ap[j + 1] == Bp[j + 1]);
			ASSERT(Anz[j] == Bnz[j]);
			if (xtype == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			else
			{
				for (; p < pend; p++)
				{
					// abs_value(xtype, Ax, Az, p, Common);
					// ASSERT(xtype == CHOLMOD_REAL); // Moved above
					// for dim==2
					if (option == 0) {
						Bx[p] = Ax[p] * h[j];
					} else if (option == 1) {
						Bx[p] = Ax[p] * h[j];
					}
					else {
						ASSERT(false);
					}
				}
			}
		}
	}
}

void my_cholmod_l_sparse_diag_mul2
(
	cholmod_sparse* A,
	double* h, 
	cholmod_sparse* B
)
{

	double s;
	double* Ax, * Az; // *W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	double* Bx, * Bz;
	Int* Bp, * Bi, * Bnz;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;

	ASSERT(A->ncol == B->ncol);
	ASSERT(A->nrow == B->nrow);

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;

	Bp = (Int*)B->p;
	Bi = (Int*)B->i;
	Bx = (double*)B->x;
	Bz = (double*)B->z;
	Bnz = (Int*)B->nz;

	packed = (Int)A->packed;
	xtype = (Int)A->xtype;


	// ASSERT(A->nz == B->nz); // nz is array not int
	ASSERT(A->packed == B->packed);
	ASSERT(A->xtype == B->xtype);

	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;

	//Int f = ncol / dim;
	//ASSERT(ncol == dim * f);
	//ASSERT(dim == 2);

	ASSERT(xtype == CHOLMOD_REAL);
	ASSERT(A->stype == 0); // A is unsymmetric
	{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			ASSERT(Ap[j] == Bp[j]);
			ASSERT(Ap[j + 1] == Bp[j + 1]);
			// ASSERT(Anz[j] == Bnz[j]); // should not check this for unpack matrix. 
			if (xtype == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			else
			{
				for (; p < pend; p++)
				{
					// abs_value(xtype, Ax, Az, p, Common);
					// ASSERT(xtype == CHOLMOD_REAL); // Moved above
					// for dim==2
					Bx[p] = Ax[p] * h[j];
					//if (j < 10 && Ai[p] < 10) {
					//	printf("updating A[%d,%d] *= %f\n", Ai[p], j, h[j]);
					//}
				}
			}
		}
	}
}

void my_cholmod_l_sparse_diag_mul2_same_pattern
(
	cholmod_sparse** mA,
	double** mb, 
	cholmod_sparse** mB,
	int num
) 
{

	double s;

	double** mAx = new double* [num];
	double** mAz = new double* [num];
	// *W;
	Int** mAp = new Int * [num];
	Int** mAi = new Int * [num];
	Int** mAnz = new Int * [num];

	Int i, j, p, pend, nrow, ncol, packed, xtype;

	double** mBx = new double* [num];
	double** mBz = new double* [num];
	// *W;
	Int** mBp = new Int * [num];
	Int** mBi = new Int * [num];
	Int** mBnz = new Int * [num];

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;

	ncol = mA[0]->ncol;
	nrow = mA[0]->nrow;

	ASSERT(mA[0]->ncol == mB[0]->ncol);
	ASSERT(mA[0]->nrow == mB[0]->nrow);

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	for (int r = 0; r < num; r++) {
		mAp[r] = (Int*)mA[r]->p;
		mAi[r] = (Int*)mA[r]->i;
		mAx[r] = (double*)mA[r]->x;
		mAz[r] = (double*)mA[r]->z;
		mAnz[r] = (Int*)mA[r]->nz;
	}

	for (int r = 0; r < num; r++) {
		mBp[r] = (Int*)mB[r]->p;
		mBi[r] = (Int*)mB[r]->i;
		mBx[r] = (double*)mB[r]->x;
		mBz[r] = (double*)mB[r]->z;
		mBnz[r] = (Int*)mB[r]->nz;
	}

	packed = (Int)mA[0]->packed;
	xtype = (Int)mA[0]->xtype;

	//for (int r = 0; r < num; r++) {
	//	// ASSERT(A->nz == B->nz); // nz is array not int
	//	ASSERT(mA[r]->packed == packed);
	//	ASSERT(mA[r]->xtype == xtype);

	//	ASSERT((Int)mA[r]->xtype == CHOLMOD_REAL);
	//	ASSERT(mA[r]->stype == 0); // A is unsymmetric, other case not implemented. 
	//}


	//for (int r = 0; r < num; r++) {
	//	// ASSERT(A->nz == B->nz); // nz is array not int
	//	ASSERT(mB[r]->packed == packed);
	//	ASSERT(mB[r]->xtype == xtype);

	//	ASSERT((Int)mB[r]->xtype == CHOLMOD_REAL);
	//	ASSERT(mB[r]->stype == 0); // A is unsymmetric, other case not implemented. 
	//}

	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;

	{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			Int* Ap = mAp[0];
			Int* Anz = mAnz[0];
			//double* Ax = mAx[0];

			Int* Bp = mBp[0];
			Int* Bnz = mBnz[0];
			//double* Bx = mBx[0];

			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			ASSERT(Ap[j] == Bp[j]);
			ASSERT(Ap[j + 1] == Bp[j + 1]);
			// ASSERT(Anz[j] == Bnz[j]); // should not check this for unpack matrix. 
			if (xtype == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			else
			{
				for (; p < pend; p++)
				{
					// abs_value(xtype, Ax, Az, p, Common);
					// ASSERT(xtype == CHOLMOD_REAL); // Moved above
					// for dim==2
					
					for (int r = 0; r < num; r++) {
						mBx[r][p] = mAx[r][p] * mb[r][j];
					}
				}
			}
		}
	}
}

void my_cholmod_l_check_same_pattern
(
	cholmod_sparse* A,
	cholmod_sparse* B
)
{

	double s;
	double* Ax, * Az; // *W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	double* Bx, * Bz;
	Int* Bp, * Bi, * Bnz;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;

	ASSERT(A->ncol == B->ncol);
	ASSERT(A->nrow == B->nrow);

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;

	Bp = (Int*)B->p;
	Bi = (Int*)B->i;
	Bx = (double*)B->x;
	Bz = (double*)B->z;
	Bnz = (Int*)B->nz;

	packed = (Int)A->packed;
	xtype = (Int)A->xtype;


	// ASSERT(A->nz == B->nz); // nz is array not int
	ASSERT(A->packed == B->packed);
	ASSERT(A->xtype == B->xtype);

	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;


	// ASSERT(false); // to see if false is optimized off.

	ASSERT(xtype == CHOLMOD_REAL);
	ASSERT(A->stype == 0); // A is unsymmetric, other case not implemented. 
	{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			ASSERT(Ap[j] == Bp[j]);
			ASSERT(Ap[j + 1] == Bp[j + 1]);
			// ASSERT(Anz[j] == Bnz[j]); // should not check this for unpack matrix.
		}
	}
}

void my_cholmod_l_assign_same_pattern
(
	cholmod_sparse* A,
	cholmod_sparse* B
)
{

	double s;
	double* Ax, * Az; // *W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	double* Bx, * Bz;
	Int* Bp, * Bi, * Bnz;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;

	ASSERT(A->ncol == B->ncol);
	ASSERT(A->nrow == B->nrow);

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;

	Bp = (Int*)B->p;
	Bi = (Int*)B->i;
	Bx = (double*)B->x;
	Bz = (double*)B->z;
	Bnz = (Int*)B->nz;

	packed = (Int)A->packed;
	xtype = (Int)A->xtype;


	// ASSERT(A->nz == B->nz); // nz is array not int
	ASSERT(A->packed == B->packed);
	ASSERT(A->xtype == B->xtype);

	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;


	// ASSERT(false); // to see if false is optimized off.

	ASSERT(xtype == CHOLMOD_REAL);
	// note we do not need ASSERT(A->stype == 0); 
	// that being said, wangyu has check the following code block will work for unsymmetric or symmetric matrices, 
	// by comparing with copy_sparse from cholmod_sparse.c
	{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			ASSERT(Ap[j] == Bp[j]);
			ASSERT(Ap[j + 1] == Bp[j + 1]);
			// ASSERT(Anz[j] == Bnz[j]); // should not check this for unpack matrix. 
			if (xtype == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			else
			{
				for (; p < pend; p++)
				{
					// abs_value(xtype, Ax, Az, p, Common);
					// ASSERT(xtype == CHOLMOD_REAL); // Moved above
					// for dim==2
					Bx[p] = Ax[p];
				}
			}
		}
	}
}

#include "cholmod_core.h"

// This returns matrix with size same as GxT * GxT', i.e. GxT * Gx
// cholmod_sparse* 
int my_cholmod_l_assmble_alap
(
	cholmod_sparse* GxT,
	cholmod_sparse* GyT,
	cholmod_sparse* GzT, // use for only dim==3
	double ** h, // h[0-2] or h[0-5] stores the 2x2 or 3x3 tensors.  
	int dim, 
	bool lower_only,
	/* --------------- */
	int*& II,
	int*& JJ,
	double*& VV
	/* --------------- */
	// cholmod_common* Common
) 
{

	const int stype = 0;

	cholmod_sparse* A = GxT;
	cholmod_sparse* B = GyT;
	cholmod_sparse* C = GzT;

	double* h00 = h[0];
	double* h01 = h[1];
	double* h10 = h01;
	double* h11 = h[2];

	double *h02, *h20, *h12, *h21, *h22;
	if (dim == 3) {
		h02 = h[3];
		h20 = h02;
		h12 = h[4];
		h21 = h12;
		h22 = h[5];
	}

	double s;
	double* Ax, * Az; // *W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	double* Bx, * Bz;
	Int* Bp, * Bi, * Bnz;

	double* Cx, * Cz;
	Int* Cp, * Ci, * Cnz;

	Int triplet_nnz;
	triplet_nnz = 0;
	
	// bool lower_only = true;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;

	ASSERT(A->ncol == B->ncol);
	ASSERT(A->nrow == B->nrow);
	if (dim == 3) {
		ASSERT(A->ncol == C->ncol);
		ASSERT(A->nrow == C->nrow);
	}

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;

	Bp = (Int*)B->p;
	Bi = (Int*)B->i;
	Bx = (double*)B->x;
	Bz = (double*)B->z;
	Bnz = (Int*)B->nz;

	if (dim == 3) {
		Cp = (Int*)C->p;
		Ci = (Int*)C->i;
		Cx = (double*)C->x;
		Cz = (double*)C->z;
		Cnz = (Int*)C->nz;
	}

	packed = (Int)A->packed;
	xtype = (Int)A->xtype;


	// ASSERT(A->nz == B->nz); // nz is array not int
	ASSERT(A->packed == B->packed);
	ASSERT(A->xtype == B->xtype);

	if (dim == 3) {
		ASSERT(A->packed == C->packed);
		ASSERT(A->xtype == C->xtype);
	}

	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;
	Int index;
	index = 0;

	//SuiteSparse_long* II;
	//SuiteSparse_long* JJ;
	//double* VV;

	ASSERT(xtype == CHOLMOD_REAL);
	ASSERT(A->stype == 0); // A is unsymmetric, other case not implemented. 
	{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		for (j = 0; j < ncol; j++) {
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);

			if (lower_only) {
				triplet_nnz += (pend - p) * (pend - p + 1) / 2;
			}
			else {
				triplet_nnz += (pend - p) * (pend - p);
			}
			
		}


		II = new int[triplet_nnz];
		JJ = new int[triplet_nnz];
		VV = new double[triplet_nnz];

		// II = new SuiteSparse_long[triplet_nnz];
		// JJ = new SuiteSparse_long[triplet_nnz];
		// VV = new double[triplet_nnz];

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			ASSERT(Ap[j] == Bp[j]);
			ASSERT(Ap[j + 1] == Bp[j + 1]);
			if (dim == 3) {
				ASSERT(Ap[j] == Cp[j]);
				ASSERT(Ap[j + 1] == Cp[j + 1]);
			}
			// ASSERT(Anz[j] == Bnz[j]); // should not check this for unpack matrix. 
			if (xtype == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			else
			{
				for (int s = p; s < pend; s++) {
					for (int t = p; t < pend; t++) {
						if (lower_only && Ai[s] >= Ai[t] || !lower_only) {

							II[index] = Ai[s];
							JJ[index] = Ai[t];
							VV[index] =
								+Ax[s] * h00[j] * Ax[t]
								+ Ax[s] * h01[j] * Bx[t]
								+ Bx[s] * h10[j] * Ax[t]
								+ Bx[s] * h11[j] * Bx[t];
							if (dim == 3) {
								VV[index] +=
									+Ax[s] * h02[j] * Cx[t]
									+ Cx[s] * h20[j] * Ax[t]
									+ Bx[s] * h12[j] * Cx[t]
									+ Cx[s] * h21[j] * Bx[t]
									+ Cx[s] * h22[j] * Cx[t];
							}
							index++;
						}
					}
				}
			}
		}
	}

	return triplet_nnz;

#if 0
	cholmod_triplet_struct T;
	T.nrow = nrow;
	T.ncol = nrow;
	T.nzmax = triplet_nnz;
	T.nnz = triplet_nnz;


	T.i = (void*)II;
	T.j = (void*)JJ;
	T.itype = CHOLMOD_LONG; /* CHOLMOD_LONG: i and j are SuiteSparse_long.  Otherwise int CHOLMOD_INT */

	T.x = (void*)(VV);
	T.z = NULL;

	T.stype = stype; /* -1 could also work.
				 Describes what parts of the matrix are considered:
		 * >0: matrix is square and symmetric.  Entries in the lower triangular
		 *     part are transposed and added to the upper triangular part when
		 *     the matrix is converted to cholmod_sparse form.
		 */


	T.xtype = CHOLMOD_REAL; /* pattern, real, complex, or zomplex */
	T.dtype = CHOLMOD_DOUBLE; /* x and z are double or float */

	 cholmod_sparse* sparse_data = cholmod_l_triplet_to_sparse(&T, 0, Common);

	 free(II);
	 free(JJ);
	 free(VV);

	 return sparse_data;
#endif
}

// This returns matrix with size same as GxT1 * GxT1', i.e. GxT * Gx
// cholmod_sparse* 
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
)
{

	const int stype = 0;

	cholmod_sparse* A = GxT1;
	cholmod_sparse* B = GyT1;
	cholmod_sparse* C = GzT1;

	cholmod_sparse* D = GxT2;
	cholmod_sparse* E = GyT2;
	cholmod_sparse* F = GzT2;

	double* h00 = h[0];
	double* h01 = h[1];
	double* h10 = h01;
	double* h11 = h[2];

	double* h02, * h20, * h12, * h21, * h22;
	if (dim == 3) {
		h02 = h[3];
		h20 = h02;
		h12 = h[4];
		h21 = h12;
		h22 = h[5];
	}

	double s;
	double* Ax, * Az; // *W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	double* Bx, * Bz;
	Int* Bp, * Bi, * Bnz;

	double* Cx, * Cz;
	Int* Cp, * Ci, * Cnz;

	double* Dx, * Dz; 
	Int* Dp, * Di, * Dnz;
	Int i2, j2, p2, pend2, nrow2, ncol2, packed2, xtype2; // though ncol2==ncol

	double* Ex, * Ez;
	Int* Ep, * Ei, * Enz;

	double* Fx, * Fz;
	Int* Fp, * Fi, * Fnz;

	Int triplet_nnz;
	triplet_nnz = 0;

	// bool lower_only = true;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;

	ASSERT(A->ncol == B->ncol);
	ASSERT(A->nrow == B->nrow);

	if (dim == 3) {
		ASSERT(A->ncol == C->ncol);
		ASSERT(A->nrow == C->nrow);
	}

	ncol2 = D->ncol;
	nrow2 = D->nrow;

	ASSERT(D->ncol == E->ncol);
	ASSERT(D->nrow == E->nrow);

	if (dim == 3) {
		ASSERT(D->ncol == F->ncol);
		ASSERT(D->nrow == F->nrow);
	}

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;

	Bp = (Int*)B->p;
	Bi = (Int*)B->i;
	Bx = (double*)B->x;
	Bz = (double*)B->z;
	Bnz = (Int*)B->nz;

	if (dim == 3) {
		Cp = (Int*)C->p;
		Ci = (Int*)C->i;
		Cx = (double*)C->x;
		Cz = (double*)C->z;
		Cnz = (Int*)C->nz;
	}

	packed = (Int)A->packed;
	xtype = (Int)A->xtype;


	// ASSERT(A->nz == B->nz); // nz is array not int
	ASSERT(A->packed == B->packed);
	ASSERT(A->xtype == B->xtype);

	if (dim == 3) {
		ASSERT(A->packed == C->packed);
		ASSERT(A->xtype == C->xtype);
	}


	Dp = (Int*)D->p;
	Di = (Int*)D->i;
	Dx = (double*)D->x;
	Dz = (double*)D->z;
	Dnz = (Int*)D->nz;

	Ep = (Int*)E->p;
	Ei = (Int*)E->i;
	Ex = (double*)E->x;
	Ez = (double*)E->z;
	Enz = (Int*)E->nz;

	if (dim == 3) {
		Fp = (Int*)F->p;
		Fi = (Int*)F->i;
		Fx = (double*)F->x;
		Fz = (double*)F->z;
		Fnz = (Int*)F->nz;
	}

	packed2 = (Int)D->packed;
	xtype2 = (Int)D->xtype;


	// ASSERT(D->nz == E->nz); // nz is array not int
	ASSERT(D->packed == E->packed);
	ASSERT(D->xtype == E->xtype);

	if (dim == 3) {
		ASSERT(D->packed == F->packed);
		ASSERT(D->xtype == F->xtype);
	}

	ASSERT(ncol == ncol2);

	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;
	Int index;
	index = 0;

	//SuiteSparse_long* II;
	//SuiteSparse_long* JJ;
	//double* VV;

	ASSERT(xtype == CHOLMOD_REAL);
	ASSERT(A->stype == 0); // A is unsymmetric, other case not implemented. 

	ASSERT(xtype2 == CHOLMOD_REAL);
	ASSERT(D->stype == 0); 

	{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		for (j = 0; j < ncol; j++) {
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);

			p2 = Dp[j];
			pend2 = (packed2) ? (Dp[j + 1]) : (p2 + Dnz[j]);

			triplet_nnz += (pend - p) * (pend2 - p2);

		}


		II = new int[triplet_nnz];
		JJ = new int[triplet_nnz];
		VV = new double[triplet_nnz];

		// II = new SuiteSparse_long[triplet_nnz];
		// JJ = new SuiteSparse_long[triplet_nnz];
		// VV = new double[triplet_nnz];

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			ASSERT(Ap[j] == Bp[j]);
			ASSERT(Ap[j + 1] == Bp[j + 1]);
			if (dim == 3) {
				ASSERT(Ap[j] == Cp[j]);
				ASSERT(Ap[j + 1] == Cp[j + 1]);
			}
			p2 = Dp[j];
			pend2 = (packed2) ? (Dp[j + 1]) : (p2 + Dnz[j]);
			ASSERT(Dp[j] == Ep[j]);
			ASSERT(Dp[j + 1] == Ep[j + 1]);
			if (dim == 3) {
				ASSERT(Dp[j] == Fp[j]);
				ASSERT(Dp[j + 1] == Fp[j + 1]);
			}
			// ASSERT(Anz[j] == Bnz[j]); // should not check this for unpack matrix. 
			if (xtype == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			if (xtype2 == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			else
			{
				for (int s = p; s < pend; s++) {
					for (int t = p2; t < pend2; t++) {
						{

							II[index] = Ai[s];
							JJ[index] = Di[t];
							VV[index] =
								+Ax[s] * h00[j] * Dx[t]
								+ Ax[s] * h01[j] * Ex[t]
								+ Bx[s] * h10[j] * Dx[t]
								+ Bx[s] * h11[j] * Ex[t];
							if (dim == 3) {
								VV[index] +=
									+Ax[s] * h02[j] * Fx[t]
									+ Cx[s] * h20[j] * Dx[t]
									+ Bx[s] * h12[j] * Fx[t]
									+ Cx[s] * h21[j] * Ex[t]
									+ Cx[s] * h22[j] * Fx[t];
							}
							index++;
						}
					}
				}
			}
		}
	}

	return triplet_nnz;

#if 0
	cholmod_triplet_struct T;
	T.nrow = nrow;
	T.ncol = nrow;
	T.nzmax = triplet_nnz;
	T.nnz = triplet_nnz;


	T.i = (void*)II;
	T.j = (void*)JJ;
	T.itype = CHOLMOD_LONG; /* CHOLMOD_LONG: i and j are SuiteSparse_long.  Otherwise int CHOLMOD_INT */

	T.x = (void*)(VV);
	T.z = NULL;

	T.stype = stype; /* -1 could also work.
				 Describes what parts of the matrix are considered:
		 * >0: matrix is square and symmetric.  Entries in the lower triangular
		 *     part are transposed and added to the upper triangular part when
		 *     the matrix is converted to cholmod_sparse form.
		 */


	T.xtype = CHOLMOD_REAL; /* pattern, real, complex, or zomplex */
	T.dtype = CHOLMOD_DOUBLE; /* x and z are double or float */

	cholmod_sparse* sparse_data = cholmod_l_triplet_to_sparse(&T, 0, Common);

	free(II);
	free(JJ);
	free(VV);

	return sparse_data;
#endif
}

void my_cholmod_l_sum_same_pattern
(
	cholmod_sparse** mA, // mA[0~num_src-1] stacks matrices to be added as B.
	cholmod_sparse* B,
	int num_src
)
{

	double s;
	double** mAx = new double* [num_src];
	double** mAz = new double* [num_src];
	// *W;
	Int** mAp = new Int * [num_src];
	Int** mAi = new Int * [num_src];
	Int** mAnz = new Int * [num_src];
	Int i, j, p, pend, nrow, ncol, packed, xtype;

	double* Bx, * Bz;
	Int* Bp, * Bi, * Bnz;

	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;
	ncol = mA[0]->ncol;
	nrow = mA[0]->nrow;

	ASSERT(mA[0]->ncol == B->ncol);
	ASSERT(mA[0]->nrow == B->nrow);

	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	for (int r = 0; r < num_src; r++) {
		mAp[r] = (Int*)mA[r]->p;
		mAi[r] = (Int*)mA[r]->i;
		mAx[r] = (double*)mA[r]->x;
		mAz[r] = (double*)mA[r]->z;
		mAnz[r] = (Int*)mA[r]->nz;
	}

	Bp = (Int*)B->p;
	Bi = (Int*)B->i;
	Bx = (double*)B->x;
	Bz = (double*)B->z;
	Bnz = (Int*)B->nz;

	packed = (Int)mA[0]->packed;
	xtype = (Int)mA[0]->xtype;

	for (int r = 0; r < num_src; r++) {
		// ASSERT(A->nz == B->nz); // nz is array not int
		ASSERT(mA[r]->packed == B->packed);
		ASSERT(mA[r]->xtype == B->xtype);

		ASSERT((Int)mA[r]->xtype == CHOLMOD_REAL);
		ASSERT(mA[r]->stype == 0); // A is unsymmetric, other case not implemented. 

	}

	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;


	// ASSERT(false); // to see if false is optimized off.

		{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < ncol; j++)
		{
			p = Bp[j];
			pend = (packed) ? (Bp[j + 1]) : (p + Bnz[j]);
			for (int r = 0; r < num_src; r++) {
				ASSERT(mAp[r][j] == Bp[j]);
				ASSERT(mAp[r][j + 1] == Bp[j + 1]);
				// ASSERT(Anz[j] == Bnz[j]); // should not check this for unpack matrix. 
			}

			if (xtype == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			else
			{
				for (; p < pend; p++)
				{
					// abs_value(xtype, Ax, Az, p, Common);
					// ASSERT(xtype == CHOLMOD_REAL); // Moved above
					// for dim==2
					double s = 0;
					for (int r = 0; r < num_src; r++) {
						s += mAx[r][p];
					}
					Bx[p] = s;
				}
			}
		}
	}
}


#define min(a,b) ((a)<(b))?((a)):((b))

void my_cholmod_l_print
(
	cholmod_sparse* A
)
{

	double s;
	double* Ax, * Az; // *W;
	Int* Ap, * Ai, * Anz;
	Int i, j, p, pend, nrow, ncol, packed, xtype;


	/* ---------------------------------------------------------------------- */
	/* check inputs */
	/* ---------------------------------------------------------------------- */

	// RETURN_IF_NULL_COMMON(EMPTY); // wangyu remove this one or it will return.
	//RETURN_IF_NULL(A, EMPTY);
	//RETURN_IF_NULL(B, EMPTY);
	//RETURN_IF_XTYPE_INVALID(A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	//RETURN_IF_XTYPE_INVALID(B, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY);
	// Common->status = CHOLMOD_OK;
	ncol = A->ncol;
	nrow = A->nrow;


	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	Ap = (Int*)A->p;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	Anz = (Int*)A->nz;


	packed = (Int)A->packed;
	xtype = (Int)A->xtype;


	/* ---------------------------------------------------------------------- */
	/* my stuff starts here */
	/* ---------------------------------------------------------------------- */
	double coeff0, coeff1, coeff2;


	// ASSERT(false); // to see if false is optimized off.

	ASSERT(xtype == CHOLMOD_REAL);
	ASSERT(A->stype == 0); // A is unsymmetric, other case not implemented. 

	{ // modified from the code from cholmod

		/* -----(------------------------------------------------------------- */
		/*  compute the 1-norm */
		/* ------------------------------------------------------------------ */

		/* 1-norm = max column sum */
		for (j = 0; j < 10; j++)
		{
			printf("Col[%d]: ", j);
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			if (xtype == CHOLMOD_PATTERN)
			{
				ASSERT(false);
			}
			else
			{

				for (; p < pend; p++)
				{
					if (Ai[p] < 10)
						printf("\t A[%d,%d]=%f", Ai[p], j, Ax[p]);
				}
			}
			printf("\n");
		}
	}
}

#include "mkl_spblas.h"

int matvec(int M, int N, int NNZ, double* csrVal, MKL_INT* csrColInd, MKL_INT* csrRowPtr, double* x, double* y, bool transpose) {

	// csrVal is array of size NNZ
	// csrColInd: size NNZ
	// csrRowPtr: size M + 1
	// x: size N
	// y: size M

	double alpha = 1.0, beta = 0.0; // no need to clear what is in y to get y=A*x, since beta=0 is used 

	// Descriptor of main sparse matrix properties
	struct matrix_descr descrA;
	// // Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA;

	// Create handle with matrix stored in CSR format
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO,
		N,  // number of rows
		M,  // number of cols
		csrRowPtr,
		csrRowPtr + 1,
		csrColInd,
		csrVal);

	// Create matrix descriptor
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

	// Analyze sparse matrix; choose proper kernels and workload balancing strategy
	mkl_sparse_optimize(csrA);

	for (int k = 0; k < 1000; k++) {
		// Compute y = alpha * A * x + beta * y
		mkl_sparse_d_mv(transpose ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE,
			alpha,
			csrA,
			descrA,
			x,
			beta,
			y);
	}
	// Release matrix handle and deallocate matrix
	mkl_sparse_destroy(csrA);
}

int spmat_csc(int M, int N, int NNZ, double* cscVal, MKL_INT* cscRowInd, MKL_INT* cscColPtr, double* x, double* y, int nColDense, bool transpose) {

	// cscVal is array of size NNZ
	// cscRowInd: size NNZ
	// cscColPtr: size N + 1
	// x: size N, or M if transposed
	// y: size M, or N if transposed

	if (false) {
		printf("last element in csrVal: %f \n", cscVal[NNZ - 1]);
		printf("last element in cscRowInd: %d \n", cscRowInd[NNZ - 1]);
		printf("last element in csrRowPtr: %d \n", cscColPtr[M]);
		if (!transpose) {
			printf("last element in x: %f \n", x[N - 1]);
			printf("last element in y: %f \n", y[M - 1]);
		}
		else {
			printf("last element in x: %f \n", x[M - 1]);
			printf("last element in y: %f \n", y[N - 1]);
		}
	}

	double alpha = 1.0, beta = 0.0; // no need to clear what is in y to get y=A*x, since beta=0 is used 

	// Descriptor of main sparse matrix properties
	struct matrix_descr descrA;
	// // Structure with sparse matrix stored in CSC format
	sparse_matrix_t       cscA;

	//sparse_status_t mkl_sparse_d_create_csc(sparse_matrix_t * A, const sparse_index_base_t
	//	indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT * cols_start, MKL_INT
	//	* cols_end, MKL_INT * row_indx, double* values);

	// Create handle with matrix stored in *CSC* format

	// sparse_status_t 
	mkl_sparse_d_create_csc(&cscA,
		SPARSE_INDEX_BASE_ZERO, // const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
		M, //const MKL_INT   rows,
		N, //const MKL_INT   cols,
		cscColPtr, //MKL_INT * cols_start,
		cscColPtr + 1, // MKL_INT * cols_end,
		cscRowInd, //MKL_INT * row_indx,
		cscVal); //double* values);

	// Create matrix descriptor
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

	// Analyze sparse matrix; choose proper kernels and workload balancing strategy
	mkl_sparse_optimize(cscA);

	// leading dimension is defined as a[i + j*lda] for column-major layout
	// and a[j + i*lda] for row-major.
	// assuming all column-major here. 
	const MKL_INT             ldx = transpose ? M : N;
	const MKL_INT             ldy = transpose ? N : M;
	const MKL_INT             columns = nColDense;

	sparse_status_t r = mkl_sparse_d_mm(transpose ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE,
		alpha,
		cscA,
		descrA,          /* sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t */
		sparse_layout_t::SPARSE_LAYOUT_COLUMN_MAJOR,         /* storage scheme for the dense matrix: C-style or Fortran-style */
		x,
		columns,
		ldx,
		beta,
		y,
		ldy);

	// Release matrix handle and deallocate matrix
	mkl_sparse_destroy(cscA);
}

int matvec_csc(int M, int N, int NNZ, double* cscVal, MKL_INT* cscRowInd, MKL_INT* cscColPtr, double* x, double* y, bool transpose) {

	// cscVal is array of size NNZ
	// cscRowInd: size NNZ
	// cscColPtr: size N + 1
	// x: size N, or M if transposed
	// y: size M, or N if transposed

	if (false) {
		printf("last element in csrVal: %f \n", cscVal[NNZ - 1]);
		printf("last element in cscRowInd: %d \n", cscRowInd[NNZ - 1]);
		printf("last element in csrRowPtr: %d \n", cscColPtr[M]);
		if (!transpose) {
			printf("last element in x: %f \n", x[N - 1]);
			printf("last element in y: %f \n", y[M - 1]);
		}
		else {
			printf("last element in x: %f \n", x[M - 1]);
			printf("last element in y: %f \n", y[N - 1]);
		}
	}

	double alpha = 1.0, beta = 0.0;

	// Descriptor of main sparse matrix properties
	struct matrix_descr descrA;
	// // Structure with sparse matrix stored in CSC format
	sparse_matrix_t       cscA;

	sparse_status_t mkl_sparse_d_create_csc(sparse_matrix_t * A, const sparse_index_base_t
		indexing, const MKL_INT rows, const MKL_INT cols, MKL_INT * cols_start, MKL_INT
		* cols_end, MKL_INT * row_indx, double* values);

	// Create handle with matrix stored in *CSC* format

	// sparse_status_t 
	mkl_sparse_d_create_csc(&cscA,
		SPARSE_INDEX_BASE_ZERO, // const sparse_index_base_t indexing, /* indexing: C-style or Fortran-style */
		M, //const MKL_INT   rows,
		N, //const MKL_INT   cols,
		cscColPtr, //MKL_INT * cols_start,
		cscColPtr + 1, // MKL_INT * cols_end,
		cscRowInd, //MKL_INT * row_indx,
		cscVal); //double* values);

	// Create matrix descriptor
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

	// Analyze sparse matrix; choose proper kernels and workload balancing strategy
	mkl_sparse_optimize(cscA);

	// leading dimension is defined as a[i + j*lda] for column-major layout
	// and a[j + i*lda] for row-major.
	// assuming all column-major here. 
	const MKL_INT             ldx = transpose ? M : N;
	const MKL_INT             ldy = transpose ? N : M;

	// Compute y = alpha * A * x + beta * y
	mkl_sparse_d_mv(transpose ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE,
		alpha,
		cscA,
		descrA,
		x,
		beta,
		y);

	// Release matrix handle and deallocate matrix
	mkl_sparse_destroy(cscA);
}


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
)
{

	double yx[8], xx[8], ax[2];

	double* Ax, * Az, * Xx, * Xz, * Yx, * Yz, * w, * Wz;
	Int* Ap, * Ai, * Anz;
	size_t nx, ny, dx, dy;
	Int packed, nrow, ncol, j, k, p, pend, kcol, i;


	/* ---------------------------------------------------------------------- */
	/* get inputs */
	/* ---------------------------------------------------------------------- */

	ny = transpose ? A->ncol : A->nrow;	/* required length of Y */
	nx = transpose ? A->nrow : A->ncol;	/* required length of X */

	nrow = A->nrow;
	ncol = A->ncol;

	Ap = (Int*)A->p;
	Anz = (Int*)A->nz;
	Ai = (Int*)A->i;
	Ax = (double*)A->x;
	Az = (double*)A->z;
	packed = A->packed;
	Xx = (double*)X->x;
	Xz = (double*)X->z;
	Yx = (double*)Y->x;
	Yz = (double*)Y->z;
	kcol = X->ncol;
	dy = Y->d;
	dx = X->d;
	w = W;
	Wz = W + 4 * nx;


	Int nrow_x = X->nrow;
	Int nrow_y = Y->nrow;
	// Int kcol = X->ncol; // already have.

	/* ---------------------------------------------------------------------- */
	/* Y = beta * Y  wangyu removed this part for nonzero beta*/
	/* ---------------------------------------------------------------------- */

	/* ---------------------------------------------------------------------- */
	/* Y += alpha * op(A) * X, where op(A)=A or A' */
	/* ---------------------------------------------------------------------- */

	Yx = (double*)Y->x;
	Yz = (double*)Y->z;

	k = 0;

	ASSERT(A->stype == 0);
	ASSERT(packed);

	const int M = A->nrow;
	const int N = A->ncol;
	const int NNZ = A->nzmax;

	double* x;
	double* y;

	// ASSERT(sizeof(INT) == sizeof(MKL_INT));

	// Do not have to clear what is in y since a beta=0 will be used in the methods. 

	if (true) {

		double* cscVal = Ax;
		MKL_INT* cscRowInd = Ai;
		MKL_INT* cscColPtr = Ap;

		x = Xx;
		y = Yx;

		if (true) {
			ASSERT(false);
			for (j = 0; j < kcol; j++)
			{
				x = &(Xx[j * nrow_x]);
				y = &(Yx[j * nrow_y]);
				matvec_csc(M, N, NNZ, cscVal, cscRowInd, cscColPtr, x, y, transpose);
			}
		} 
		else {
			spmat_csc(M, N, NNZ, cscVal, cscRowInd, cscColPtr, x, y, kcol, transpose); // matmat();
			// use kcol instead of nCol since A could be transposed. 
		}
	}
	else {

		ASSERT(false); // TODO

		double* csrVal = Ax;
		MKL_INT* csrColInd = Ai;
		MKL_INT* csrRowPtr = Ap;

	}

#if 0
	{
		for (j = 0; j < ncol; j++)
		{
			p = Ap[j];
			pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
			if (xtype == CHOLMOD_PATTERN)
			{
				s = pend - p;
			}
			else
			{
				s = 0;
				for (; p < pend; p++)
				{
					s += abs_value(xtype, Ax, Az, p, Common); // i.e.  Ax[p] for real
				}
			}
			sum += s;
		}
	}

	{
		/* xj0 = alpha [0] * x [j     ] ; */
		/* xj1 = alpha [0] * x [j+  dx] ; */
		/* xj2 = alpha [0] * x [j+2*dx] ; */
		/* xj3 = alpha [0] * x [j+3*dx] ; */
		MULT(xx, xz, 0, alpha, alphaz, 0, Xx, Xz, j);
		MULT(xx, xz, 1, alpha, alphaz, 0, Xx, Xz, j + dx);
		MULT(xx, xz, 2, alpha, alphaz, 0, Xx, Xz, j + 2 * dx);
		MULT(xx, xz, 3, alpha, alphaz, 0, Xx, Xz, j + 3 * dx);

		p = Ap[j];
		pend = (packed) ? (Ap[j + 1]) : (p + Anz[j]);
		for (; p < pend; p++)
		{
			i = Ai[p];
			/* aij = Ax [p] ; */
			ASSIGN(ax, az, 0, Ax, Az, p);

			/* y [i     ] += aij * xj0 ; */
			/* y [i+  dy] += aij * xj1 ; */
			/* y [i+2*dy] += aij * xj2 ; */
			/* y [i+3*dy] += aij * xj3 ; */
			MULTADD(Yx, Yz, i, ax, az, 0, xx, xz, 0);
			MULTADD(Yx, Yz, i + dy, ax, az, 0, xx, xz, 1);
			MULTADD(Yx, Yz, i + 2 * dy, ax, az, 0, xx, xz, 2);
			MULTADD(Yx, Yz, i + 3 * dy, ax, az, 0, xx, xz, 3);
		}
	}
#endif
}
