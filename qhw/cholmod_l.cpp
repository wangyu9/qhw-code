/* This one is modified from Tim Davis's Demo/cholmod_l_demo */


/* Read in a matrix from a file, and use CHOLMOD to solve Ax=b if A is
 * symmetric, or (AA'+beta*I)x=b otherwise.  The file format is a simple
 * triplet format, compatible with most files in the Matrix Market format.
 * See cholmod_read.c for more details.  The readhb.f program reads a
 * Harwell/Boeing matrix (excluding element-types) and converts it into the
 * form needed by this program.  reade.f reads a matrix in Harwell/Boeing
 * finite-element form.
 *
 * The matrix is assumed to be positive definite (a supernodal LL' or simplicial
 * LDL' factorization is used).
 *
 * Requires the Core, Cholesky, MatrixOps, and Check Modules.
 * Optionally uses the Partition and Supernodal Modules.
 * Does not use the Modify Module.
 *
 * See cholmod_simple.c for a simpler demo program.
 *
 * SuiteSparse_long is normally defined as long, except for WIN64.
 */

#include "cholmod_l.h"
#define NTRIALS 100

int initLHS(cholmod_sparse* A, cholmod_common * cm)
{
    SuiteSparse_long i;
    // 

    int prefer_zomplex;
    prefer_zomplex = cm->prefer_zomplex;

    if (prefer_zomplex && A->xtype == CHOLMOD_COMPLEX)
    {
        /* Convert to zomplex, just for testing.  In a zomplex matrix,
           the real and imaginary parts are in separate arrays.  MATLAB
           uses zomplex matrix exclusively. */
        double *Ax = (double *) A->x ;// wangyu add this implicit conversion and a few ones later. 
        SuiteSparse_long nz = cholmod_l_nnz (A, cm) ;
        printf ("nz: %ld\n", nz) ;
        double *Ax2 = (double *) cholmod_l_malloc (nz, sizeof (double), cm) ;
        double *Az2 = (double *) cholmod_l_malloc (nz, sizeof (double), cm) ;
        for (i = 0 ; i < nz ; i++)
        {
            Ax2 [i] = Ax [2*i  ] ;
            Az2 [i] = Ax [2*i+1] ;
        }
        cholmod_l_free (A->nzmax, 2*sizeof(double), Ax, cm) ;
        A->x = Ax2 ;
        A->z = Az2 ;
        A->xtype = CHOLMOD_ZOMPLEX ;
        /* cm->print = 5 ; */
    }

    cholmod_l_print_sparse (A, "A", cm) ;

#if 0
    if ( 0 ) {
      // scale diagonal
      printf ("\n\n     SCALING DIAGONAL   \n\n");
      
      // create diagonal
      printf ("%ld,%ld,%d\n", A->nrow, A->ncol, A->xtype );

      cholmod_sparse *D = cholmod_l_speye (A->nrow, A->ncol, A->xtype, cm );
      printf ("sparse done \n");
      cholmod_l_print_sparse (D, "D", cm);

      D->stype = 1;
      cholmod_l_print_sparse (D, "D", cm);

      double alpha[2];
      double beta[2];
      alpha[0] = 1.0;
      alpha[1] = 1.0;
      beta[0] = 1.0e9; // 9 works, 467doesn't
      beta[1] = 1.0e0;

      cholmod_sparse *C = cholmod_l_add (A, D, alpha, beta, 1, 0, cm );
      cholmod_l_print_sparse (C, "C", cm);

      A = C;

    }
#endif

    if (A->nrow > A->ncol)
    {
	    /* Transpose A so that A'A+beta*I will be factorized instead */
	    cholmod_sparse *C = cholmod_l_transpose (A, 2, cm) ;
	    cholmod_l_free_sparse (&A, cm) ;
	    A = C ;
	    printf ("transposing input matrix\n") ;
    }
    return 0;
}

int initRHS(cholmod_dense* B) {
    SuiteSparse_long i = 0;
    int xtype = B->xtype;
    SuiteSparse_long n = B->nrow;
    /* ---------------------------------------------------------------------- */
    /* create an arbitrary right-hand-side */
    /* ---------------------------------------------------------------------- */

    double* Bx, * Bz;

    // n = A->nrow;
    
    Bx = (double*)B->x;
    Bz = (double*)B->z;

#if GHS
    {
        /* b = A*ones(n,1), used by Gould, Hu, and Scott in their experiments */
        cholmod_dense* X0;
        X0 = cholmod_l_ones(A->ncol, 1, xtype, cm);
        cholmod_l_sdmult(A, 0, one, zero, X0, B, cm);
        cholmod_l_free_dense(&X0, cm);
    }
#else
    if (xtype == CHOLMOD_REAL)
    {
        /* real case */
        for (i = 0; i < n; i++)
        {
            double x = n;
            Bx[i] = 1 + i / x;
        }
    }
    else if (xtype == CHOLMOD_COMPLEX)
    {
        /* complex case */
        for (i = 0; i < n; i++)
        {
            double x = n;
            Bx[2 * i] = 1 + i / x;		/* real part of B(i) */
            Bx[2 * i + 1] = (x / 2 - i) / (3 * x);	/* imag part of B(i) */
        }
    }
    else /* (xtype == CHOLMOD_ZOMPLEX) */
    {
        /* zomplex case */
        for (i = 0; i < n; i++)
        {
            double x = n;
            Bx[i] = 1 + i / x;		/* real part of B(i) */
            Bz[i] = (x / 2 - i) / (3 * x);	/* imag part of B(i) */
        }
    }

#endif

    return 0;
}

int linSolve_ori(cholmod_sparse* A, cholmod_dense* B, cholmod_dense* XXX, cholmod_common* cm) {

    double* Bx, * Bz;

    Bx = (double*)B->x;
    Bz = (double*)B->z;

    //
    cholmod_dense* X = NULL;

    double resid[4], t, ta, tf, ts[3], tot, bnorm, xnorm, anorm, rnorm, fl,
        anz, axbnorm, rnorm2, resid2, rcond;

    double* Xx, * Xz;

    cholmod_dense* W;
    
    double one[2], zero[2], minusone[2], beta[2], xlnz;

    cholmod_factor* L;

    cholmod_dense* R = NULL;
    double* Rx, * Rz;
    
    SuiteSparse_long i, n, isize, xsize, ordering, xtype, s, ss, lnz;
    int trial, method, L_is_super;

    int nmethods;

    ts[0] = 0.;
    ts[1] = 0.;
    ts[2] = 0.;

    xtype = A->xtype;

    n = B->nrow; // critical to initialize n here. 

    /* ---------------------------------------------------------------------- */
    /* create basic scalars */
    /* ---------------------------------------------------------------------- */

    zero[0] = 0;
    zero[1] = 0;
    one[0] = 1;
    one[1] = 0;
    minusone[0] = -1;
    minusone[1] = 0;
    beta[0] = 1e-6;
    beta[1] = 0;

    anorm = 1;
#ifndef NMATRIXOPS
    anorm = cholmod_l_norm_sparse(A, 0, cm);
    printf("norm (A,inf) = %g\n", anorm);
    printf("norm (A,1)   = %g\n", cholmod_l_norm_sparse(A, 1, cm));
#endif


    cholmod_l_print_dense (B, "B", cm) ;
    bnorm = 1 ;
#ifndef NMATRIXOPS
    bnorm = cholmod_l_norm_dense (B, 0, cm) ;	/* max norm */
    printf ("bnorm %g\n", bnorm) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    t = CPUTIME ;
    L = cholmod_l_analyze (A, cm) ;
    ta = CPUTIME - t ;
    ta = MAX (ta, 0) ;

    printf ("Analyze: flop %g lnz %g\n", cm->fl, cm->lnz) ;

    if (A->stype == 0)
    {
	    printf ("Factorizing A*A'+beta*I\n") ;
	    t = CPUTIME ;
	    cholmod_l_factorize_p (A, beta, NULL, 0, L, cm) ;
	    tf = CPUTIME - t ;
	    tf = MAX (tf, 0) ;
    }
    else
    {
        const int NUM_FACTOR = 1;
	    printf ("Factorizing A %d times\n", NUM_FACTOR);
	    t = CPUTIME ;
        for(int iter=0; iter<NUM_FACTOR; iter++){
	        cholmod_l_factorize (A, L, cm) ;
        }
	    tf = CPUTIME - t ;
	    tf = MAX (tf, 0) ;
    }

    cholmod_l_print_factor (L, "L", cm) ;

    /* determine the # of integers's and reals's in L.  See cholmod_free */
    if (L->is_super)
    {
	    s = L->nsuper + 1 ;
	    xsize = L->xsize ;
	    ss = L->ssize ;
	    isize =
	        n	/* L->Perm */
	        + n	/* L->ColCount, nz in each column of 'pure' L */
	        + s	/* L->pi, column pointers for L->s */
	        + s	/* L->px, column pointers for L->x */
	        + s	/* L->super, starting column index of each supernode */
	        + ss ;	/* L->s, the pattern of the supernodes */
    }
    else
    {
	    /* this space can increase if you change parameters to their non-
	     * default values (cm->final_pack, for example). */
	    lnz = L->nzmax ;
	    xsize = lnz ;
	    isize =
	        n	/* L->Perm */
	        + n	/* L->ColCount, nz in each column of 'pure' L */
	        + n+1	/* L->p, column pointers */
	        + lnz	/* L->i, integer row indices */
	        + n	/* L->nz, nz in each column of L */
	        + n+2	/* L->next, link list */
	        + n+2 ;	/* L->prev, link list */
    }

    /* solve with Bset will change L from simplicial to supernodal */
    rcond = cholmod_l_rcond (L, cm) ;
    L_is_super = L->is_super ;

    /* ---------------------------------------------------------------------- */
    /* solve */
    /* ---------------------------------------------------------------------- */

	/*
    if (n >= 1000)
    {
        nmethods = 1 ;
    }
    else if (xtype == CHOLMOD_ZOMPLEX)
    {
        nmethods = 2 ;
    }
    else
    {
        nmethods = 3 ;
    }*/
	nmethods = 3; // wangyu
    printf ("nmethods: %d\n", nmethods) ;

    for (method = 0 ; method <= nmethods ; method++)
    {
        double x = n ;
        resid [method] = -1 ;       /* not yet computed */

        if (method == 0)
        {
            /* basic solve, just once */
            t = CPUTIME ;
            X = cholmod_l_solve (CHOLMOD_A, L, B, cm) ;
            ts [0] = CPUTIME - t ;
            ts [0] = MAX (ts [0], 0) ;
        }
        else if (method == 1)
        {
            /* basic solve, many times, but keep the last one */
            t = CPUTIME ;
            for (trial = 0 ; trial < NTRIALS ; trial++)
            {
                cholmod_l_free_dense (&X, cm) ;
                Bx [0] = 1 + trial / x ;        /* tweak B each iteration */
                X = cholmod_l_solve (CHOLMOD_A, L, B, cm) ;
            }
            ts [1] = CPUTIME - t ;
            ts [1] = MAX (ts [1], 0) / NTRIALS ;
        }
        else if (method == 2)
        {
            /* solve with reused workspace */
            cholmod_dense *Ywork = NULL, *Ework = NULL ;
            cholmod_l_free_dense (&X, cm) ;

            t = CPUTIME ;
            for (trial = 0 ; trial < NTRIALS ; trial++)
            {
                Bx [0] = 1 + trial / x ;        /* tweak B each iteration */
                cholmod_l_solve2 (CHOLMOD_A, L, B, NULL, &X, NULL,
                    &Ywork, &Ework, cm) ;
            }
            cholmod_l_free_dense (&Ywork, cm) ;
            cholmod_l_free_dense (&Ework, cm) ;
            ts [2] = CPUTIME - t ;
            ts [2] = MAX (ts [2], 0) / NTRIALS ;
            
        }
        else
        {
            /* solve with reused workspace and sparse Bset */
            cholmod_dense *Ywork = NULL, *Ework = NULL ;
            cholmod_dense *X2 = NULL, *B2 = NULL ;
            cholmod_sparse *Bset, *Xset = NULL ;
            SuiteSparse_long *Bsetp, *Bseti, *Xsetp, *Xseti, xlen, j, k, *Lnz ;
            double *X1x, *X2x, *B2x, err ;
            FILE *timelog = fopen ("timelog.m", "w") ;
            if (timelog) fprintf (timelog, "results = [\n") ;

            B2 = cholmod_l_zeros (n, 1, xtype, cm) ;
            B2x = (double *) B2->x ;

            Bset = cholmod_l_allocate_sparse (n, 1, 1, FALSE, TRUE, 0,
                CHOLMOD_PATTERN, cm) ;
            Bsetp = (long int *) Bset->p ;
            Bseti = (long int *) Bset->i ;
            Bsetp [0] = 0 ;     /* nnz(B) is 1 (it can be anything) */
            Bsetp [1] = 1 ;
            resid [3] = 0 ;

            for (i = 0 ; i < MIN (100,n) ; i++)
            {
                /* B (i) is nonzero, all other entries are ignored
                   (implied to be zero) */
                Bseti [0] = i ;
                if (xtype == CHOLMOD_REAL)
                {
                    B2x [i] = 3.1 * i + 0.9 ;
                }
                else /* (xtype == CHOLMOD_COMPLEX) */
                {
                    B2x [2*i  ] = i + 0.042 ;
                    B2x [2*i+1] = i - 92.7 ;
                }

                /* first get the entire solution, to compare against */
                cholmod_l_solve2 (CHOLMOD_A, L, B2, NULL, &X, NULL,
                    &Ywork, &Ework, cm) ;

                /* now get the sparse solutions; this will change L from
                   supernodal to simplicial */

                if (i == 0)
                {
                    /* first solve can be slower because it has to allocate
                       space for X2, Xset, etc, and change L.
                       So don't time it */
                    cholmod_l_solve2 (CHOLMOD_A, L, B2, Bset, &X2, &Xset,
                        &Ywork, &Ework, cm) ;
                }

                t = CPUTIME ;
                for (trial = 0 ; trial < NTRIALS ; trial++)
                {
                    /* solve Ax=b but only to get x(i).
                       b is all zero except for b(i).
                       This takes O(xlen) time */
                    cholmod_l_solve2 (CHOLMOD_A, L, B2, Bset, &X2, &Xset,
                        &Ywork, &Ework, cm) ;
                }
                t = CPUTIME - t ;
                t = MAX (t, 0) / NTRIALS ;

                /* check the solution and log the time */
                Xsetp = (long int *) Xset->p ;
                Xseti = (long int *) Xset->i ;
                xlen = Xsetp [1] ;
                X1x = (double *) X->x ;
                X2x = (double *) X2->x ;
                Lnz = (long int *) L->nz ;

                if (xtype == CHOLMOD_REAL)
                {
                    fl = 2 * xlen ;
                    for (k = 0 ; k < xlen ; k++)
                    {
                        j = Xseti [k] ;
                        fl += 4 * Lnz [j] ;
                        err = X1x [j] - X2x [j] ;
                        err = ABS (err) ;
                        resid [3] = MAX (resid [3], err) ;
                    }
                }
                else /* (xtype == CHOLMOD_COMPLEX) */
                {
                    fl = 16 * xlen ;
                    for (k = 0 ; k < xlen ; k++)
                    {
                        j = Xseti [k] ;
                        fl += 16 * Lnz [j] ;
                        err = X1x [2*j  ] - X2x [2*j  ] ;
                        err = ABS (err) ;
                        resid [3] = MAX (resid [3], err) ;
                        err = X1x [2*j+1] - X2x [2*j+1] ;
                        err = ABS (err) ;
                        resid [3] = MAX (resid [3], err) ;
                    }
                }

                if (timelog) fprintf (timelog, "%g %g %g %g\n",
                    (double) i, (double) xlen, fl, t);

                /* clear B for the next test */
                if (xtype == CHOLMOD_REAL)
                {
                    B2x [i] = 0 ;
                }
                else /* (xtype == CHOLMOD_COMPLEX) */
                {
                    B2x [2*i  ] = 0 ;
                    B2x [2*i+1] = 0 ;
                }
            }

            if (timelog)
            {
                fprintf (timelog, "] ; resid = %g ;\n", resid [3]) ;
                fprintf (timelog, "lnz = %g ;\n", cm->lnz) ;
                fprintf (timelog, "t = %g ;   %% dense solve time\n", ts [2]) ;
                fclose (timelog) ;
            }

#ifndef NMATRIXOPS
            resid [3] = resid [3] / cholmod_l_norm_dense (X, 1, cm) ;
#endif

            cholmod_l_free_dense (&Ywork, cm) ;
            cholmod_l_free_dense (&Ework, cm) ;
            cholmod_l_free_dense (&X2, cm) ;
            cholmod_l_free_dense (&B2, cm) ;
            cholmod_l_free_sparse (&Xset, cm) ;
            cholmod_l_free_sparse (&Bset, cm) ;
        }

        /* ------------------------------------------------------------------ */
        /* compute the residual */
        /* ------------------------------------------------------------------ */

        if (method < 3)
        {
#ifndef NMATRIXOPS
            if (A->stype == 0)
            {
                /* (AA'+beta*I)x=b is the linear system that was solved */
                /* W = A'*X */
                W = cholmod_l_allocate_dense (A->ncol, 1, A->ncol, xtype, cm) ;
                cholmod_l_sdmult (A, 2, one, zero, X, W, cm) ;
                /* R = B - beta*X */
                cholmod_l_free_dense (&R, cm) ;
                R = cholmod_l_zeros (n, 1, xtype, cm) ;
                Rx = (double *) R->x ;
                Rz = (double *) R->z ;
                Xx = (double *) X->x ;
                Xz = (double *) X->z ;
                if (xtype == CHOLMOD_REAL)
                {
                    for (i = 0 ; i < n ; i++)
                    {
                        Rx [i] = Bx [i] - beta [0] * Xx [i] ;
                    }
                }
                else if (xtype == CHOLMOD_COMPLEX)
                {
                    /* complex case */
                    for (i = 0 ; i < n ; i++)
                    {
                        Rx [2*i  ] = Bx [2*i  ] - beta [0] * Xx [2*i  ] ;
                        Rx [2*i+1] = Bx [2*i+1] - beta [1] * Xx [2*i+1] ;
                    }
                }
                else /* (xtype == CHOLMOD_ZOMPLEX) */
                {
                    /* zomplex case */
                    for (i = 0 ; i < n ; i++)
                    {
                        Rx [i] = Bx [i] - beta [0] * Xx [i] ;
                        Rz [i] = Bz [i] - beta [1] * Xz [i] ;
                    }
                }

                /* R = A*W - R */
                cholmod_l_sdmult (A, 0, one, minusone, W, R, cm) ;
                cholmod_l_free_dense (&W, cm) ;
            }
            else
            {
                /* Ax=b was factorized and solved, R = B-A*X */
                cholmod_l_free_dense (&R, cm) ;
                R = cholmod_l_copy_dense (B, cm) ;
                cholmod_l_sdmult (A, 0, minusone, one, X, R, cm) ;
            }
            rnorm = cholmod_l_norm_dense (R, 0, cm) ;	    /* max abs. entry */
            xnorm = cholmod_l_norm_dense (X, 0, cm) ;	    /* max abs. entry */

            axbnorm = (anorm * xnorm + bnorm + ((n == 0) ? 1 : 0)) ;
            resid [method] = rnorm / axbnorm ;
#else
            printf ("residual not computed (requires CHOLMOD/MatrixOps)\n") ;
#endif
        }
    }

    tot = ta + tf + ts [0] ;

    /* ---------------------------------------------------------------------- */
    /* iterative refinement (real symmetric case only) */
    /* ---------------------------------------------------------------------- */

    double norm_before = cholmod_l_norm_dense(X, 2, cm);
    printf("Norm before refinement: %f\n", norm_before);

    resid2 = -1 ;
    const int iter_refine = 1; // wangyu added 
    if (iter_refine) {
#ifndef NMATRIXOPS
        if (A->stype != 0 && A->xtype == CHOLMOD_REAL)
        {
            cholmod_dense* R2;

            /* R2 = A\(B-A*X) */
            R2 = cholmod_l_solve(CHOLMOD_A, L, R, cm);
            /* compute X = X + A\(B-A*X) */
            Xx = (double*)X->x;
            Rx = (double*)R2->x;
            for (i = 0; i < n; i++)
            {
                Xx[i] = Xx[i] + Rx[i];
            }
            cholmod_l_free_dense(&R2, cm);
            cholmod_l_free_dense(&R, cm);

            /* compute the new residual, R = B-A*X */
            cholmod_l_free_dense(&R, cm);
            R = cholmod_l_copy_dense(B, cm);
            cholmod_l_sdmult(A, 0, minusone, one, X, R, cm);
            rnorm2 = cholmod_l_norm_dense(R, 0, cm);
            resid2 = rnorm2 / axbnorm;
        }
#endif
    }
    cholmod_l_free_dense (&R, cm) ;

    double norm_after = cholmod_l_norm_dense(X, 2, cm);
    printf("Norm after refinement: %f\n", norm_after);

    /* ---------------------------------------------------------------------- */
    /* print results */
    /* ---------------------------------------------------------------------- */

    anz = cm->anz ;
    for (i = 0 ; i < CHOLMOD_MAXMETHODS ; i++)
    /* for (i = 4 ; i < 3 ; i++) */
    {
	fl = cm->method [i].fl ;
	xlnz = cm->method [i].lnz ;
	cm->method [i].fl = -1 ;
	cm->method [i].lnz = -1 ;
	ordering = cm->method [i].ordering ;
	if (fl >= 0)
	{
	    printf ("Ordering: ") ;
	    if (ordering == CHOLMOD_POSTORDERED) printf ("postordered ") ;
	    if (ordering == CHOLMOD_NATURAL)     printf ("natural ") ;
	    if (ordering == CHOLMOD_GIVEN)	 printf ("user    ") ;
	    if (ordering == CHOLMOD_AMD)	 printf ("AMD     ") ;
	    if (ordering == CHOLMOD_METIS)	 printf ("METIS   ") ;
	    if (ordering == CHOLMOD_NESDIS)      printf ("NESDIS  ") ;
	    if (xlnz > 0)
	    {
		printf ("fl/lnz %10.1f", fl / xlnz) ;
	    }
	    if (anz > 0)
	    {
		printf ("  lnz/anz %10.1f", xlnz / anz) ;
	    }
	    printf ("\n") ;
	}
    }

    printf ("ints in L: %15.0f, doubles in L: %15.0f\n",
        (double) isize, (double) xsize) ;
    printf ("factor flops %g nnz(L) %15.0f (w/no amalgamation)\n",
	    cm->fl, cm->lnz) ;
    if (A->stype == 0)
    {
	printf ("nnz(A):    %15.0f\n", cm->anz) ;
    }
    else
    {
	printf ("nnz(A*A'): %15.0f\n", cm->anz) ;
    }
    if (cm->lnz > 0)
    {
	printf ("flops / nnz(L):  %8.1f\n", cm->fl / cm->lnz) ;
    }
    if (anz > 0)
    {
	printf ("nnz(L) / nnz(A): %8.1f\n", cm->lnz / cm->anz) ;
    }
    printf ("analyze cputime:  %12.4f\n", ta) ;
    printf ("factor  cputime:   %12.4f mflop: %8.1f\n", tf,
	(tf == 0) ? 0 : (1e-6*cm->fl / tf)) ;
    printf ("solve   cputime:   %12.4f mflop: %8.1f\n", ts [0],
	(ts [0] == 0) ? 0 : (1e-6*4*cm->lnz / ts [0])) ;
    printf ("overall cputime:   %12.4f mflop: %8.1f\n", 
	    tot, (tot == 0) ? 0 : (1e-6 * (cm->fl + 4 * cm->lnz) / tot)) ;
    printf ("solve   cputime:   %12.4f mflop: %8.1f (%d trials)\n", ts [1],
	(ts [1] == 0) ? 0 : (1e-6*4*cm->lnz / ts [1]), NTRIALS) ;
    printf ("solve2  cputime:   %12.4f mflop: %8.1f (%d trials)\n", ts [2],
	(ts [2] == 0) ? 0 : (1e-6*4*cm->lnz / ts [2]), NTRIALS) ;
    printf ("peak memory usage: %12.0f (MB)\n",
	    (double) (cm->memory_usage) / 1048576.) ;
    printf ("residual (|Ax-b|/(|A||x|+|b|)): ") ;
    for (method = 0 ; method <= nmethods ; method++)
    {
        printf ("%8.2e ", resid [method]) ;
    }
    printf ("\n") ;
    if (resid2 >= 0)
    {
	printf ("residual %8.1e (|Ax-b|/(|A||x|+|b|))"
		" after iterative refinement\n", resid2) ;
    }
    printf ("rcond    %8.1e\n\n", rcond) ;

    if (L_is_super)
    {
        cholmod_l_gpu_stats (cm) ;
    }

    //if (false) { // TODO: wangyu remove this, but causing memory leakage issues for now. Fix it later. 
        cholmod_l_free_factor(&L, cm);
    //}

    cholmod_l_copy_dense2(X, XXX, cm); // wangyu added this. 
    
    /* ---------------------------------------------------------------------- */
    /* free matrices and finish CHOLMOD */
    /* ---------------------------------------------------------------------- */

    return 0;
}


#define Long SuiteSparse_long

inline void mexErrMsgTxt(const char* a) {
    printf("%s", a);
}

inline void mexWarnMsgTxt(const char* a) {
    printf("%s", a);
}

#define mexPrintf printf

Long mxGetM(cholmod_sparse* A) {
    return A->nrow;
}

Long mxGetN(cholmod_sparse* A) {
    return A->ncol;
}

bool mxIsSparse(cholmod_sparse* A) {
    return true;
}

#define DBL_EPSILON 1e-10

// Factor A and store results in L. 
// Here is critical to pass the reference to the pointer of cholmod_factor
// otherwise the value assigned to L will not be passed to the upper level. 
int symbolic_factorize(cholmod_sparse* A, cholmod_factor*& L, cholmod_common* cm)
{
    double dummy = 0, * p; // rcond,
    cholmod_sparse Amatrix; // , Bspmatrix, * Bs, * Xs; // * A,
    // cholmod_dense Bmatrix, * X; // , * B;

    // cholmod_common Common, * cm;
    Long n, ordering, k, * Perm; // B_is_sparse, 

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */
    /* ---------------------------------------------------------------------- */

    // cm = &Common;
    // cholmod_l_start(cm);
    // sputil_config(SPUMONI, cm);

    /* There is no supernodal LDL'.  If cm->final_ll = FALSE (the default), then
     * this mexFunction will use a simplicial LDL' when flops/lnz < 40, and a
     * supernodal LL' otherwise.  This may give suprising results to the MATLAB
     * user, so always perform an LL' factorization by setting cm->final_ll
     * to TRUE. */

    cm->final_ll = TRUE;
    cm->quick_return_if_not_posdef = TRUE;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (false)// if (nargout > 2 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt("usage: [x,rcond] = cholmod2 (A,b,ordering)");
    }
    n = mxGetM(A); // mxGetM(pargin[0]);
    if (n != mxGetN(A)) // (!mxIsSparse(pargin[0]) || (n != mxGetN(pargin[0])))
    {
        mexErrMsgTxt("A must be square and sparse");
    }
    if (n != mxGetM(A)) // (n != mxGetM(pargin[1]))
    {
        mexErrMsgTxt("# of rows of A and B must match");
    }

    /* get sparse matrix A.  Use triu(A) only. */
    // A = sputil_get_sparse(pargin[0], &Amatrix, &dummy, 1);

    /* get sparse or dense matrix B */
#if 0
    B = NULL;
    Bs = NULL;
    B_is_sparse = mxIsSparse(pargin[1]);
    if (B_is_sparse)
    {
        /* get sparse matrix B (unsymmetric) */
        Bs = sputil_get_sparse(pargin[1], &Bspmatrix, &dummy, 0);
    }
    else
    {
        /* get dense matrix B */
        B = sputil_get_dense(pargin[1], &Bmatrix, &dummy);
    }

    /* get the ordering option */
    if (nargin < 3)
    {
        /* use default ordering */
        ordering = -1;
    }
    else
    {
        /* use a non-default option */
        ordering = mxGetScalar(pargin[2]);
    }
#else
    ordering = -1;
#endif

    p = NULL;
    Perm = NULL;

    if (ordering == 0)
    {
        /* natural ordering */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_NATURAL;
        cm->postorder = FALSE;
    }
    else if (ordering == -1)
    {
        /* default strategy ... nothing to change */
    }
    else if (ordering == -2)
    {
        /* default strategy, but with NESDIS in place of METIS */
        cm->default_nesdis = TRUE;
    }
    else if (ordering == -3)
    {
        /* use AMD only */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_AMD;
        cm->postorder = TRUE;
    }
    else if (ordering == -4)
    {
        /* use METIS only */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_METIS;
        cm->postorder = TRUE;
    }
    else if (ordering == -5)
    {
        /* use NESDIS only */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_NESDIS;
        cm->postorder = TRUE;
    }
    else if (ordering == -6)
    {
        /* natural ordering, but with etree postordering */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_NATURAL;
        cm->postorder = TRUE;
    }
    else if (ordering == -7)
    {
        /* always try both AMD and METIS, and pick the best */
        cm->nmethods = 2;
        cm->method[0].ordering = CHOLMOD_AMD;
        cm->method[1].ordering = CHOLMOD_METIS;
        cm->postorder = TRUE;
    }
#if 0
    else if (ordering >= 1)
    {
        /* assume the 3rd argument is a user-provided permutation of 1:n */
        if (mxGetNumberOfElements(pargin[2]) != n)
        {
            mexErrMsgTxt("invalid input permutation");
        }
        /* copy from double to integer, and convert to 0-based */
        p = mxGetPr(pargin[2]);
        Perm = cholmod_l_malloc(n, sizeof(Long), cm);
        for (k = 0; k < n; k++)
        {
            Perm[k] = p[k] - 1;
        }
        /* check the permutation */
        if (!cholmod_l_check_perm(Perm, n, n, cm))
        {
            mexErrMsgTxt("invalid input permutation");
        }
        /* use only the given permutation */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_GIVEN;
        cm->postorder = FALSE;
    }
#endif
    else
    {
        mexErrMsgTxt("invalid ordering option");
    }

    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    L = cholmod_l_analyze_p(A, Perm, NULL, 0, cm);
    cholmod_l_free(n, sizeof(Long), Perm, cm);
}

int numerical_factorize(cholmod_sparse* A, cholmod_factor* L, cholmod_common* cm)
{
    cholmod_l_factorize(A, L, cm);

    double rcond = cholmod_l_rcond(L, cm);

    if (rcond == 0)
    {
        mexWarnMsgTxt("Matrix is indefinite or singular to working precision");
    }
    else if (rcond < DBL_EPSILON)
    {
        mexWarnMsgTxt("Matrix is close to singular or badly scaled.");
        mexPrintf("         Results may be inaccurate. RCOND = %g.\n", rcond);
    }

}

// Here it is ok not to pass reference of XXX, since cholmod_l_copy_dense2 
// is used to copy content of cholmod_dense.
int solveLLT(cholmod_factor* L, cholmod_dense* B, cholmod_dense* XXX, cholmod_common* cm) {

	cholmod_sparse Bspmatrix, * Bs, * Xs;
	cholmod_dense Bmatrix, * X; // , * B;
	Long B_is_sparse;


	/* ---------------------------------------------------------------------- */
	/* solve and return solution to MATLAB */
	/* ---------------------------------------------------------------------- */
#if 0

	if (B_is_sparse)
	{
		/* solve AX=B with sparse X and B; return sparse X to MATLAB */
		Xs = cholmod_l_spsolve(CHOLMOD_A, L, Bs, cm);
		pargout[0] = sputil_put_sparse(&Xs, cm);
	}
	else
	{
		/* solve AX=B with dense X and B; return dense X to MATLAB */
		X = cholmod_l_solve(CHOLMOD_A, L, B, cm);
		pargout[0] = sputil_put_dense(&X, cm);
	}
	/* return statistics, if requested */
	if (nargout > 1)
	{
		pargout[1] = mxCreateDoubleMatrix(1, 5, mxREAL);
		p = mxGetPr(pargout[1]);
		p[0] = rcond;
		p[1] = L->ordering;
		p[2] = cm->lnz;
		p[3] = cm->fl;
		p[4] = cm->memory_usage / 1048576.;
	}
#else
	X = cholmod_l_solve(CHOLMOD_A, L, B, cm);
	cholmod_l_copy_dense2(X, XXX, cm);
#endif
}

int linSolve(cholmod_sparse* A, cholmod_dense* B, cholmod_dense* XXX, cholmod_common* cm)
{
    cholmod_factor* L = NULL;

    symbolic_factorize(A, L, cm);
    numerical_factorize(A, L, cm);
    solveLLT(L, B, XXX, cm);

	cholmod_l_free_factor(&L, cm);

	// cholmod_l_finish(cm);
	// cholmod_l_print_common(" ", cm);
}

// This is the one works by modifying the MATLAB wrapper. 
int linSolve2(cholmod_sparse* A, cholmod_dense* B, cholmod_dense* XXX, cholmod_common* cm)
{
    double dummy = 0, rcond, * p;
    cholmod_sparse Amatrix, Bspmatrix,  * Bs, * Xs; // * A,
    cholmod_dense Bmatrix, * X; // , * B;
    cholmod_factor* L;
    // cholmod_common Common, * cm;
    Long n, B_is_sparse, ordering, k, * Perm;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */
    /* ---------------------------------------------------------------------- */

    // cm = &Common;
    // cholmod_l_start(cm);
    // sputil_config(SPUMONI, cm);

    /* There is no supernodal LDL'.  If cm->final_ll = FALSE (the default), then
     * this mexFunction will use a simplicial LDL' when flops/lnz < 40, and a
     * supernodal LL' otherwise.  This may give suprising results to the MATLAB
     * user, so always perform an LL' factorization by setting cm->final_ll
     * to TRUE. */

    cm->final_ll = TRUE;
    cm->quick_return_if_not_posdef = TRUE;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (false)// if (nargout > 2 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt("usage: [x,rcond] = cholmod2 (A,b,ordering)");
    }
    n = mxGetM(A); // mxGetM(pargin[0]);
    if (n != mxGetN(A)) // (!mxIsSparse(pargin[0]) || (n != mxGetN(pargin[0])))
    {
        mexErrMsgTxt("A must be square and sparse");
    }
    if (n != mxGetM(A)) // (n != mxGetM(pargin[1]))
    {
        mexErrMsgTxt("# of rows of A and B must match");
    }

    /* get sparse matrix A.  Use triu(A) only. */
    // A = sputil_get_sparse(pargin[0], &Amatrix, &dummy, 1);

    /* get sparse or dense matrix B */
#if 0
    B = NULL;
    Bs = NULL;
    B_is_sparse = mxIsSparse(pargin[1]);
    if (B_is_sparse)
    {
        /* get sparse matrix B (unsymmetric) */
        Bs = sputil_get_sparse(pargin[1], &Bspmatrix, &dummy, 0);
    }
    else
    {
        /* get dense matrix B */
        B = sputil_get_dense(pargin[1], &Bmatrix, &dummy);
    }

    /* get the ordering option */
    if (nargin < 3)
    {
        /* use default ordering */
        ordering = -1;
    }
    else
    {
        /* use a non-default option */
        ordering = mxGetScalar(pargin[2]);
    }
#else
    ordering = -1;
#endif

    p = NULL;
    Perm = NULL;

    if (ordering == 0)
    {
        /* natural ordering */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_NATURAL;
        cm->postorder = FALSE;
    }
    else if (ordering == -1)
    {
        /* default strategy ... nothing to change */
    }
    else if (ordering == -2)
    {
        /* default strategy, but with NESDIS in place of METIS */
        cm->default_nesdis = TRUE;
    }
    else if (ordering == -3)
    {
        /* use AMD only */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_AMD;
        cm->postorder = TRUE;
    }
    else if (ordering == -4)
    {
        /* use METIS only */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_METIS;
        cm->postorder = TRUE;
    }
    else if (ordering == -5)
    {
        /* use NESDIS only */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_NESDIS;
        cm->postorder = TRUE;
    }
    else if (ordering == -6)
    {
        /* natural ordering, but with etree postordering */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_NATURAL;
        cm->postorder = TRUE;
    }
    else if (ordering == -7)
    {
        /* always try both AMD and METIS, and pick the best */
        cm->nmethods = 2;
        cm->method[0].ordering = CHOLMOD_AMD;
        cm->method[1].ordering = CHOLMOD_METIS;
        cm->postorder = TRUE;
    }
#if 0
    else if (ordering >= 1)
    {
        /* assume the 3rd argument is a user-provided permutation of 1:n */
        if (mxGetNumberOfElements(pargin[2]) != n)
        {
            mexErrMsgTxt("invalid input permutation");
        }
        /* copy from double to integer, and convert to 0-based */
        p = mxGetPr(pargin[2]);
        Perm = cholmod_l_malloc(n, sizeof(Long), cm);
        for (k = 0; k < n; k++)
        {
            Perm[k] = p[k] - 1;
        }
        /* check the permutation */
        if (!cholmod_l_check_perm(Perm, n, n, cm))
        {
            mexErrMsgTxt("invalid input permutation");
        }
        /* use only the given permutation */
        cm->nmethods = 1;
        cm->method[0].ordering = CHOLMOD_GIVEN;
        cm->postorder = FALSE;
    }
#endif
    else
    {
        mexErrMsgTxt("invalid ordering option");
    }

    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    L = cholmod_l_analyze_p(A, Perm, NULL, 0, cm);
    cholmod_l_free(n, sizeof(Long), Perm, cm);
    cholmod_l_factorize(A, L, cm);

    rcond = cholmod_l_rcond(L, cm);

    if (rcond == 0)
    {
        mexWarnMsgTxt("Matrix is indefinite or singular to working precision");
    }
    else if (rcond < DBL_EPSILON)
    {
        mexWarnMsgTxt("Matrix is close to singular or badly scaled.");
        mexPrintf("         Results may be inaccurate. RCOND = %g.\n", rcond);
    }

    /* ---------------------------------------------------------------------- */
    /* solve and return solution to MATLAB */
    /* ---------------------------------------------------------------------- */
#if 0

    if (B_is_sparse)
    {
        /* solve AX=B with sparse X and B; return sparse X to MATLAB */
        Xs = cholmod_l_spsolve(CHOLMOD_A, L, Bs, cm);
        pargout[0] = sputil_put_sparse(&Xs, cm);
    }
    else
    {
        /* solve AX=B with dense X and B; return dense X to MATLAB */
        X = cholmod_l_solve(CHOLMOD_A, L, B, cm);
        pargout[0] = sputil_put_dense(&X, cm);
    }
    /* return statistics, if requested */
    if (nargout > 1)
    {
        pargout[1] = mxCreateDoubleMatrix(1, 5, mxREAL);
        p = mxGetPr(pargout[1]);
        p[0] = rcond;
        p[1] = L->ordering;
        p[2] = cm->lnz;
        p[3] = cm->fl;
        p[4] = cm->memory_usage / 1048576.;
    }
#else
    X = cholmod_l_solve(CHOLMOD_A, L, B, cm);
    cholmod_l_copy_dense2(X, XXX, cm);
#endif
    cholmod_l_free_factor(&L, cm);
    // cholmod_l_finish(cm);
    // cholmod_l_print_common(" ", cm);

}
