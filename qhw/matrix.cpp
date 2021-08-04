#include "matrix.h"

#include "cholmod_demo.h"
#include "cholmod_l.h"
#include "cholmod_plus.h"

#define INT_MATRIX_DATA_TYPE CHOLMOD_REAL

#include "mkl_for_cholmod.h"

#include <stdio.h>
#include <assert.h>

// #define DO_NOT_USE_MKL

#ifndef DO_NOT_USE_MKL
#include "mkl.h"
#include "mkl_vml.h"
#endif

#include "timer.h"

#define RAISE_ERROR(msg)  fprintf(stderr, (msg));

/* halt if an error occurs */
void simple_error_handler(int status, const char* file, int line,
    const char* message)
{
    printf("cholmod error: file: %s line: %d status: %d: %s\n",
        file, line, status, message);
}

cholmod_common* Begin() {
    cholmod_common * cm = new cholmod_common();

    cholmod_l_start(cm);
    CHOLMOD_FUNCTION_DEFAULTS;     /* just for testing (not required) */

    /* cm->useGPU = 1; */
    cm->prefer_zomplex = 0;

    /* use default parameter settings, except for the error handler.  This
        * demo program terminates if an error occurs (out of memory, not positive
        * definite, ...).  It makes the demo program simpler (no need to check
        * CHOLMOD error conditions).  This non-default parameter setting has no
        * effect on performance. */
    cm->error_handler = simple_error_handler; // my_handler;

    return cm;
}

cholmod_common* cm_default() {
    cholmod_common* cm = new cholmod_common();

    cholmod_l_start(cm);
    CHOLMOD_FUNCTION_DEFAULTS;     /* just for testing (not required) */

    /* cm->useGPU = 1; */
    cm->prefer_zomplex = 0;

    /* use default parameter settings, except for the error handler.  This
        * demo program terminates if an error occurs (out of memory, not positive
        * definite, ...).  It makes the demo program simpler (no need to check
        * CHOLMOD error conditions).  This non-default parameter setting has no
        * effect on performance. */
    cm->error_handler = simple_error_handler; // my_handler;

    return cm;
}

int configure_solve(cholmod_common* cm) {
    cm->nmethods = 1;
    cm->method[0].ordering = CHOLMOD_AMD; // CHOLMOD_NESDIS; //CHOLMOD_AMD; // CHOLMOD_NATURAL; //CHOLMOD_METIS ;
    cm->postorder = TRUE;

    cm->nmethods = 4;
    cm->method[0].ordering = CHOLMOD_AMD;
    cm->method[1].ordering = CHOLMOD_NESDIS; //CHOLMOD_AMD; // 
    cm->method[2].ordering = CHOLMOD_NATURAL; //
    cm->method[3].ordering = CHOLMOD_METIS;
    cm->postorder = TRUE;
    return 0; 
}

int End(cholmod_common* cm) {
    cholmod_l_finish(cm);
    return 0;
}

/* ff is a global variable so that it can be closed by my_handler */
FILE* ff;

/* halt if an error occurs */
static void my_handler(int status, const char* file, int line,
    const char* message)
{
    printf("cholmod error: file: %s line: %d status: %d: %s\n",
        file, line, status, message);
    if (status < 0)
    {
        if (ff != NULL) fclose(ff);
        exit(0);
    }
}

Sparse::Sparse(const int* I, const int* J, const double* V, const int m, const int n, const int nnz, const int stype, cholmod_common* cm) {
    this->cm = cm;
    this->L = NULL;

    // TODO: optional, here copying 
#if 1
    SuiteSparse_long* II = new SuiteSparse_long[nnz];
    for (int i = 0; i < nnz; i++) {
        II[i] = I[i];
    }

	SuiteSparse_long* JJ = new SuiteSparse_long[nnz];
	for (int i = 0; i < nnz; i++) {
		JJ[i] = J[i];
	}
    // note II and JJ are freed later. 
#endif

    cholmod_triplet_struct T;
    T.nrow = m;
    T.ncol = n;
    T.nzmax = nnz;
    T.nnz = nnz;

#if 1
    T.i = (void*)II;
    T.j = (void*)JJ;
    T.itype = CHOLMOD_LONG; /* CHOLMOD_LONG: i and j are SuiteSparse_long.  Otherwise int CHOLMOD_INT */
#else
	T.i = (void*)I;
	T.j = (void*)J;
    T.itype = CHOLMOD_INT;
#endif

    T.x = (void*)(V);
    T.z = NULL;

    T.stype = stype; /* -1 could also work.
                 Describes what parts of the matrix are considered:
         * >0: matrix is square and symmetric.  Entries in the lower triangular
         *     part are transposed and added to the upper triangular part when
         *     the matrix is converted to cholmod_sparse form.
         */

    
    T.xtype = CHOLMOD_REAL; /* pattern, real, complex, or zomplex */
    T.dtype = CHOLMOD_DOUBLE; /* x and z are double or float */

    // TODO: free existing memory that is associated with sparse_data

    sparse_data = cholmod_l_triplet_to_sparse(&T, 0, this->cm); 
    // Put an 0 there just following the example, TODO: figure out the reason. 
#if 1
    free(II);
    free(JJ); 
#endif
}

Sparse::Sparse(cholmod_common* cm) {
    this->cm = cm;
    this->L = NULL;

    /* This does not work. Probably some other initialization not done. 
    size_t nrow = 1;	
    size_t ncol = 1;	
    size_t nzmax = 1;	
    int sorted = true;	// TRUE if columns of A sorted, FALSE otherwise 
    int packed = true;	// TRUE if A will be packed, FALSE otherwise 
	int stype = 1; 		// See cholmod_core.h. stype=1: symmetric and triu(A) is stored
    int xtype = CHOLMOD_REAL;	
    sparse_data = cholmod_l_allocate_sparse(nrow, ncol, nzmax, sorted, packed, stype, xtype, cm);
    */
    
    sparse_data = cholmod_l_speye(1, 1, CHOLMOD_REAL, this->cm);
}

Sparse::~Sparse() {
    deconstruct();
}

void Sparse::deconstruct() {
	if (L != NULL) {
		cholmod_l_free_factor(&L, cm);
	}
	L = NULL;
	cholmod_l_free_sparse(&sparse_data, cm); // note it asks for cholmod_sparse **
    sparse_data = NULL;
}

void Sparse::deconstruct_keep_factor() {
	cholmod_l_free_sparse(&sparse_data, cm); // note it asks for cholmod_sparse **
	sparse_data = NULL;
}

// difference between copy constructor vs assignment operator is critical! 
// See e.g. https://www.geeksforgeeks.org/copy-constructor-vs-assignment-operator-in-c/

Sparse::Sparse(const Sparse& s) {
	// nor should it call: if (L != NULL) { cholmod_l_free_factor(&L, cm); }
	L = NULL;
    // should not do this as sparse_data is not inited yet!: cholmod_l_free_sparse(&sparse_data, cm);
    cm = cm_default(); // this causes seg fault: *cm = *s.cm;
    sparse_data = cholmod_l_copy_sparse(s.sparse_data, cm);
}

Sparse& Sparse::operator = (const Sparse& s) {
    Sparse::assign(s, *this);
    return *this;
}

void Sparse::assign(const Sparse& src, Sparse& dest, int request_type) {
	if (dest.L != NULL) { cholmod_l_free_factor(&dest.L, dest.cm); }
	dest.L = NULL; // existing factorization is destroyed if exists. 
    cholmod_l_free_sparse(&dest.sparse_data, dest.cm);
    dest.cm = cm_default(); // this causes seg fault: *dest.cm = *src.cm;
    if (false) { // old implemetation that does not support requested type. 
        dest.sparse_data = cholmod_l_copy_sparse(src.sparse_data, dest.cm);
    }
    else {
        int stype = (request_type == COPY_STYPE_OF_SRC_MATRIX) ? src.sparse_data->stype : request_type;
        int mode = 1;		/* >0: numerical, 0: pattern, <0: pattern (no diag) */

        dest.sparse_data = cholmod_l_copy(src.sparse_data, stype, mode, dest.cm); 
    }
}



void Sparse::assign_value_same_pattern(const Sparse& src, Sparse& dest) {
    int nnz = src.sparse_data->nzmax;
    if (nnz != dest.sparse_data->nzmax || 
        src.nrow() != dest.nrow() ||
        src.ncol() != dest.ncol()
        ) {
        printf("Error: assign_value_same_pattern(): the src and dest sparse matrix has different pattern, %d(src)!=%d(dest) !\n",
            src.sparse_data->nzmax, dest.sparse_data->nzmax);
        return;
    }

    if (true) { // naive implementation. 
        my_cholmod_l_assign_same_pattern(src.sparse_data, dest.sparse_data);
    }
    else {
        ;
    }
}

void Sparse::sum_same_pattern(const Sparse** src, Sparse& dest, int num_src) {
    int nnz = dest.sparse_data->nzmax;
    for (int i = 0; i < num_src; i++) {
        if (nnz != (*src[i]).sparse_data->nzmax ||
                (*src[i]).nrow() != dest.nrow() ||
                (*src[i]).ncol() != dest.ncol()
            ) {
            printf("Error: sum_same_pattern(): the src and dest sparse matrix has different pattern, %d(src)!=%d(dest) !\n",
                (*src[i]).sparse_data->nzmax, dest.sparse_data->nzmax);
            return;
        }
    }

    cholmod_sparse** matrices = new cholmod_sparse*[num_src];

    for (int i = 0; i < num_src; i++) {
        matrices[i] = (*src[i]).sparse_data;
    } 

    if (true) { // naive implementation. 
        my_cholmod_l_sum_same_pattern(matrices, dest.sparse_data, num_src);
    }
    else {
        ;
    }

    free(matrices);
}

Sparse Sparse::assemble_lap(const Sparse& GxT, const Sparse& GyT, const Sparse& GzT, const Dense& au, int dim) {
    const int f = GxT.ncol();
    const int n = GxT.nrow();
    const int mcdim = (dim == 2) ? 3 : 6;
    assert(au.nrow() == mcdim * f);
    // Sparse A;
    double* hh = au.head();
    double* h[6] = { 
        hh, &hh[f], &hh[f * 2],  
        (dim == 2) ? NULL : &hh[f * 3], 
        (dim == 2) ? NULL : &hh[f * 4], 
        (dim == 2) ? NULL : &hh[f * 5] };
    int* II = NULL;
    int* JJ = NULL;
    double* VV = NULL;

    Sparse R;

    if (true) {
        int nnz = my_cholmod_l_assmble_alap(
            GxT.sparse_data,
            GyT.sparse_data,
            GzT.sparse_data,
            h, dim,
            true, 
            II, JJ, VV // cm_default()
        );
        int stype = 1; // // 1: symmetric and use triu

        return Sparse(II, JJ, VV, n, n, nnz, stype, cm_default());
        // TODO: free II, JJ, VV
            
    }
    else {

        int nnz = my_cholmod_l_assmble_alap(
            GxT.sparse_data,
            GyT.sparse_data,
            GzT.sparse_data,
            h, dim,
            false,
            II, JJ, VV // cm_default()
        );
        int stype = 0; // // 1: symmetric and use triu
        // Critical: use unsymmetric type 0 here
        // since the version of II, JJ, VV from my_cholmod_l_assmble_alap
        // keep both lower and upper entries. 
        Sparse A = Sparse(II, JJ, VV, n, n, nnz, stype, cm_default());

        int request_type = 1;
        Sparse::assign(A, R, request_type);
    }

    free(II);
    free(JJ);
    free(VV);

    return R;
}

Sparse Sparse::assemble_lap_off_diag(const Sparse& GxT, const Sparse& GyT, const Sparse& GzT, const Sparse& GxT2, const Sparse& GyT2, const Sparse& GzT2, const Dense& au, int dim) {
    const int f = GxT.ncol();
    const int n = GxT.nrow();
    const int n2 = GxT2.nrow();
    const int mcdim = (dim == 2) ? 3 : 6;
    assert(au.nrow() == mcdim * f);
    // Sparse A;
    double* hh = au.head();
    double* h[6] = {
        hh, &hh[f], &hh[f * 2],
        (dim == 2) ? NULL : &hh[f * 3],
        (dim == 2) ? NULL : &hh[f * 4],
        (dim == 2) ? NULL : &hh[f * 5] };
    int* II = NULL;
    int* JJ = NULL;
    double* VV = NULL;

    Sparse R;

    {

        int nnz = my_cholmod_l_assmble_alap_off_diag(
            GxT.sparse_data,
            GyT.sparse_data,
            GzT.sparse_data,
            GxT2.sparse_data,
            GyT2.sparse_data,
            GzT2.sparse_data,
            h, dim,
            II, JJ, VV // cm_default()
        );
        int stype = 0; // // 1: symmetric and use triu
        // Critical: use unsymmetric type 0 here
        // since the version of II, JJ, VV from my_cholmod_l_assmble_alap
        // keep both lower and upper entries. 
        R = Sparse(II, JJ, VV, n, n2, nnz, stype, cm_default());

        // int request_type = 1;
        // Sparse::assign(A, R, request_type);
    }

    delete[] II;
    delete[] JJ;
    delete[] VV;

    return R;
}


void sp_mul_same_pattern(const Sparse** src0, const Sparse** src1, Sparse** dest, int num_src)
{
    int nnz0 = (*src0[0]).sparse_data->nzmax;
    int nrow0 = (*src0[0]).nrow();
    int ncol0 = (*src0[0]).ncol();
    for (int i = 0; i < num_src; i++) {
        if (nnz0 != (*src0[i]).sparse_data->nzmax ||
            (*src0[i]).nrow() != nrow0 ||
            (*src0[i]).ncol() != ncol0
            ) {
            printf("Error: sp_mul_same_pattern(): the src0 sparse matrices have different pattern\n");
            return;
        }
    }

    int nnz1 = (*src1[0]).sparse_data->nzmax;
    int nrow1 = (*src1[0]).nrow();
    int ncol1 = (*src1[0]).ncol();
    for (int i = 0; i < num_src; i++) {
        if (nnz1 != (*src1[i]).sparse_data->nzmax ||
            (*src1[i]).nrow() != nrow1 ||
            (*src1[i]).ncol() != ncol1
            ) {
            printf("Error: sp_mul_same_pattern(): the src1 sparse matrices have different pattern\n");
            return;
        }
    }

    cholmod_sparse** matrices0 = new cholmod_sparse * [num_src];
    cholmod_sparse** matrices1 = new cholmod_sparse * [num_src];

    cholmod_sparse** result_matrices = new cholmod_sparse * [num_src];

    for (int i = 0; i < num_src; i++) {
        matrices0[i] = (*src0[i]).sparse_data;
        matrices1[i] = (*src1[i]).sparse_data;
        result_matrices[i] = (*dest[i]).sparse_data;
    }

    if (true) { // naive implementation. 
        ; // my_cholmod_l_sum_same_pattern(matrices0, matrices1, result_matrices, num_src);
    }
    else {
        ;
    }

    free(matrices0);
    free(matrices1);
    free(result_matrices);
}


double Sparse::norm(int type_of_norm) const {
    double norm = cholmod_l_norm_sparse(sparse_data, type_of_norm, cm); /* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
    return norm;
}

double Sparse::my_norm(int type_of_norm) const {
	double norm = my_cholmod_l_norm_sparse(sparse_data, type_of_norm, cm); /* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
	return norm;
}

double Sparse::pnorm(int power) const {
	double sum = my_cholmod_l_pnorm_sparse(sparse_data, power, cm); /* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
	return sum;
}

void Sparse::init_ori(const std::string filename, const int prefer_zomplex) {

    FILE* f;
    // cholmod_sparse* A;

    int ver[3];
    // int prefer_zomplex;

    /* ---------------------------------------------------------------------- */
    /* get the file containing the input matrix */
    /* ---------------------------------------------------------------------- */

    ff = NULL;
    // prefer_zomplex = 0;
    // if (argc > 1)
    {
        if ((f = fopen(filename.c_str(), "r")) == NULL)
        {
            my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
                "unable to open file");
        }
        ff = f;
        //prefer_zomplex = (argc > 2);
    }
    /*
    else
    {
        if ((f = fopen(filename.c_str(), "r")) == NULL)
        {
            my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
                "unable to open file");
            f = stdin;
        }
        ff = f;
        prefer_zomplex = (argc > 2);
    }
    */

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */
    /* ---------------------------------------------------------------------- */

    
    cholmod_l_start(cm);
    CHOLMOD_FUNCTION_DEFAULTS;     /* just for testing (not required) */

    /* cm->useGPU = 1; */
    cm->prefer_zomplex = prefer_zomplex;

    /* use default parameter settings, except for the error handler.  This
        * demo program terminates if an error occurs (out of memory, not positive
        * definite, ...).  It makes the demo program simpler (no need to check
        * CHOLMOD error conditions).  This non-default parameter setting has no
        * effect on performance. */
    cm->error_handler = my_handler;

    /* Note that CHOLMOD will do a supernodal LL' or a simplicial LDL' by
        * default, automatically selecting the latter if flop/nnz(L) < 40. */

        // wangyu:
    cm->nmethods = 1;
    cm->method[0].ordering = CHOLMOD_AMD; // CHOLMOD_NESDIS; //CHOLMOD_AMD; // CHOLMOD_NATURAL; //CHOLMOD_METIS ;
    cm->postorder = TRUE;


    cm->nmethods = 4;
    cm->method[0].ordering = CHOLMOD_AMD;
    cm->method[1].ordering = CHOLMOD_NESDIS; //CHOLMOD_AMD; // 
    cm->method[2].ordering = CHOLMOD_NATURAL; //
    cm->method[3].ordering = CHOLMOD_METIS;
    cm->postorder = TRUE;

    /* ---------------------------------------------------------------------- */
    /* read in a matrix */
    /* ---------------------------------------------------------------------- */

    printf("\n---------------------------------- cholmod_l_demo:\n");
    cholmod_l_version(ver);
    printf("cholmod version %d.%d.%d\n", ver[0], ver[1], ver[2]);
    SuiteSparse_version(ver);
    printf("SuiteSparse version %d.%d.%d\n", ver[0], ver[1], ver[2]);
    sparse_data = cholmod_l_read_sparse(f, cm);
    if (ff != NULL)
    {
        fclose(ff);
        ff = NULL;
    }

    initLHS(sparse_data, cm);
}

void Sparse::init_ori2(const std::string filename, const int prefer_zomplex) {

    configure_solve(cm);

    FILE* f;

    // int ver[3];

    ff = NULL;
    {
        if ((f = fopen(filename.c_str(), "r")) == NULL)
        {
            my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
                "unable to open file");
        }
        ff = f;
    }

    sparse_data = cholmod_l_read_sparse(f, cm);
    if (ff != NULL)
    {
        fclose(ff);
        ff = NULL;
    }

    initLHS(sparse_data, cm);
}

/*
int Sparse::initFromFile(const std::string filename) {
    FILE* f;
    if ((f = fopen(filename.c_str(), "r")) == NULL)
    {
        printf( //my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
            "unable to open file");
        return -1;
    }
    FILE* ff = f;

    A = cholmod_l_read_sparse(f, cm);

    fclose(ff);
    return 0;
}
*/

int Sparse::read(const std::string filename) {

    FILE* f;
    // cholmod_sparse* A;

    // int ver[3];
    // int prefer_zomplex;

    /* ---------------------------------------------------------------------- */
    /* get the file containing the input matrix */
    /* ---------------------------------------------------------------------- */

    ff = NULL;
    // prefer_zomplex = 0;
    // if (argc > 1)
    {
        if ((f = fopen(filename.c_str(), "r")) == NULL)
        {
            my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
                "unable to open file");
        }
        ff = f;
        //prefer_zomplex = (argc > 2);
    }

    deconstruct();

    // int prefer = sym_prefer;
    sparse_data = cholmod_l_read_sparse(f, cm);
    if (ff != NULL)
    {
        fclose(ff);
        ff = NULL;
    }
    const int t = type();
    assert(t == INT_MATRIX_DATA_TYPE);
    return 0;
}


int Sparse::write(const std::string filename) {

    FILE* f;

    ff = NULL;
    {
        if ((f = fopen(filename.c_str(), "w")) == NULL)
        {
            my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
                "unable to create file");
        }
        ff = f;
        return 1;
    }

    cholmod_l_write_sparse(f, this->sparse_data, NULL, "", this->cm);
    if (ff != NULL)
    {
        fclose(ff);
        ff = NULL;
    }

    return 0;
}

void Sparse::print() {
    my_cholmod_l_print(this->sparse_data);
}

Sparse Sparse::transposed() const {
	Sparse T;
    T.deconstruct();
    T.cm = cm_default(); // this causes seg fault: *T.cm = *this->cm;

    assert(this->type() == CHOLMOD_REAL);
    int types = 1; // array transpose. 
    if (true) {
        T.sparse_data = cholmod_l_transpose(this->sparse_data, types, T.cm);
    }
    else {
        T.sparse_data = cholmod_l_speye(1, 1, CHOLMOD_REAL, T.cm);
        int ncol = this->ncol();
        SuiteSparse_long* ind = new SuiteSparse_long[ncol];
        for (int i = 0; i < ncol; i++) {
            ind[i] = i;
        }
        cholmod_l_transpose_unsym
        (
            /* ---- input ---- */
            this->sparse_data, // cholmod_sparse * A,	/* matrix to transpose */
            types, // int values,		/* 0: pattern, 1: array transpose, 2: conj. transpose */
            NULL, // SuiteSparse_long* Perm,		/* size nrow, if present (can be NULL) */
            ind, //SuiteSparse_long* fset,		/* subset of 0:(A->ncol)-1 */
            ncol, //size_t fsize,	/* size of fset */
            /* ---- output --- */
            T.sparse_data, // cholmod_sparse * F,	/* F = A', A(:,f)', or A(p,f)' */
            /* --------------- */
            this->cm
        );
    }
    
    return T;
}

void Sparse::symbolic_factor() {
    symbolic_factorize(sparse_data, L, cm);
}

void Sparse::numerical_factor() {
	/* The analysis can be re-used simply by calling this
	*routine a second time with another matrix. A must have the same nonzero
		* pattern as that passed to cholmod_analyze. */
	numerical_factorize(sparse_data, L, cm);
}

void Sparse::solve_with_factor(const Dense& B, Dense& X) const {
    solveLLT(L, B.dense_data, X.dense_data, cm);
}

Dense Sparse::solve_with_factor(const Dense& B) const {
	Dense X(B.nrow(), B.ncol());
	Sparse::solve_with_factor(B, X);
	return X;
}

void Sparse::solve(const Dense& B, Dense& X) const {
    linSolve(sparse_data, B.dense_data, X.dense_data, cm);
}

Dense Sparse::solve(const Dense& B) const {
    Dense X(B.nrow(), B.ncol());
    this->solve(B, X);
    return X;
}

Sparse Sparse::operator + (const Sparse& B) const {
	return Sparse::add(*this, B, 1.0, 1.0);
}

Sparse Sparse::operator - (const Sparse& B) const {
	return Sparse::subtract(*this, B);
}

Sparse Sparse::operator * (const double& m) const {
	return Sparse::add(*this, *this, m, 0.0); // this may be improved. 
}

Dense Sparse::mul(const Dense& X) const {
    Dense Y(this->nrow(), X.ncol());
    this->mul(X, Y);
    return Y;
}

Dense Sparse::operator * (const Dense& X) const { // Keep it same as mul. 
    Dense Y(this->nrow(), X.ncol());
    this->mul(X, Y);
    return Y;
}

void Sparse::mul(const Dense& X, Dense& Y) const {
    // Y = A*X

    // TODO: optional, if Y has wrong size, destroy it and re-create. 

    // Check Y has the right size, otherwise cholmod_l_sdmult will complain. 
    assert(X.nrow() == this->ncol());
    assert(Y.nrow() == this->nrow());
    assert(Y.ncol() == X.ncol());
    
    bool use_cholmod = true;
    if (use_cholmod){

        /* Sparse matrix times dense matrix */
        // cholmod_sdmult: Y = alpha * (A * X) + beta * Y or alpha * (A'*X) + beta*Y */

        /* ---- input ---- */
        cholmod_sparse * A = sparse_data;
        int transpose = 0;	/* use A if 0, or A' otherwise */
        double alpha[2] = { 1, 0 };   /* scale factor for A, the complex part has to be 0. */
        double beta[2] = { 0, 0 };    /* scale factor for Y */

        cholmod_l_sdmult(A, transpose, alpha, beta,
                        X.dense_data, Y.dense_data, X.cm);
    }
    else {
        assert(false);
    }

    return;
}

void Sparse::mul2(const Dense& X, Dense& Y) const {
	// Y = A*X

	// TODO: optional, if Y has wrong size, destroy it and re-create. 

	// Check Y has the right size, otherwise cholmod_l_sdmult will complain. 
	assert(X.nrow() == this->ncol());
	assert(Y.nrow() == this->nrow());
	assert(Y.ncol() == X.ncol());

	/* Sparse matrix times dense matrix */
	// cholmod_sdmult: Y = alpha * (A * X) + beta * Y or alpha * (A'*X) + beta*Y */

	/* ---- input ---- */
	cholmod_sparse* A = sparse_data;
	int transpose = 0;	/* use A if 0, or A' otherwise */
	double alpha[2] = { 1, 0 };   /* scale factor for A, the complex part has to be 0. */
	double beta[2] = { 0, 0 };    /* scale factor for Y */

    //my_cholmod_l_sdmult_omp(A, transpose, alpha, beta,
	//	X.dense_data, Y.dense_data, X.cm);

    {
        assert(A->stype == 0); 
        // makes sure the matrix is stored as unsym, though it may be sym in fact
        // so the API agrees with the MKL one. 
        my_cholmod_l_sdmult_mkl(A, transpose, alpha, beta,
                X.dense_data, Y.dense_data, X.cm);
    }

	return;
}

void Sparse::mul(const Sparse& B, Sparse& Y, int requested_stype, bool keep_factor) const {
    int stype = requested_stype;		// 0 is unsymmetric, requested stype of C 
    int values = 1;		/* TRUE: do numerical values, FALSE: pattern only */
    int sorted; /* if TRUE then return C with sorted columns */

    int ori_nrow = 0;
    int ori_ncol = 0;
    if (keep_factor) {
        sorted = Y.sparse_data->sorted;
        ori_nrow = Y.sparse_data->nrow;
        ori_ncol = Y.sparse_data->ncol;
        assert(requested_stype==Y.sparse_data->stype);
        Y.deconstruct_keep_factor();
    }
    else {
        sorted = this->sparse_data->sorted;
        Y.deconstruct();
    }
    // *Y.cm = *this->cm; // Critical: should not do this, since result may no longer be symmetric etc. 
    Y.sparse_data = cholmod_l_ssmult(this->sparse_data, B.sparse_data, stype, values, sorted, this->cm);

    if (keep_factor)
    {
        assert(Y.sparse_data->nrow == ori_nrow);
        assert(Y.sparse_data->ncol == ori_ncol);
    }
}

Sparse Sparse::mul(const Sparse& B, int requested_stype) const {
    Sparse Y;
    this->mul(B, Y, requested_stype);
    return Y;
}

Sparse Sparse::operator * (const Sparse& B) const {
    return this->mul(B);
}

int Sparse::nrow() const {
//SuiteSparse_long to int
    return (int)sparse_data->nrow;
}

int Sparse::ncol() const {
    //SuiteSparse_long to int
    return (int)sparse_data->ncol;
}

int Sparse::type() const {
    return sparse_data->xtype;
}

//int Sparse::nz_at_col(int iCol) const {
//    return (int) ((int*)sparse_data->nz)[iCol];
//}

#if 0
#include "Core/cholmod_triplet.c"
#endif 

void Sparse::check_same_pattern(const Sparse& A, const Sparse& B) {
    my_cholmod_l_check_same_pattern(A.sparse_data, B.sparse_data);
}

void Sparse::add(const Sparse& A, const Sparse& B, Sparse& R, const double alpha, const double beta) {
    assert(A.nrow() == B.nrow());
    assert(A.ncol() == B.ncol());

    // TODO: optional, this can be optimized. 
    //double one[2] = { 1.0, 0.0 }; 
    double coeffA[2] = { alpha, 0.0};
    double coeffB[2] = { beta, 0.0 };

    cholmod_l_free_sparse(&R.sparse_data, R.cm);
    R.sparse_data = cholmod_l_add(A.sparse_data, B.sparse_data, coeffA, coeffB, 1, 1, A.cm);
}

Sparse Sparse::add(const Sparse& A, const Sparse& B, const double alpha, const double beta) {
    Sparse R = Sparse();
	Sparse::add(A, B, R, alpha, beta);
	return R;
}

void Sparse::subtract(const Sparse& A, const Sparse& B, Sparse& R) {
	add(A, B, R, 1.0, -1.0);
	return;
}

Sparse Sparse::subtract(const Sparse& A, const Sparse& B) {
	return Sparse::add(A, B, 1.0, -1.0);
}

Sparse Sparse::Diag(const Dense& diagm) {
    if (diagm.ncol() != 1 || diagm.nrow()==0) {
        RAISE_ERROR("Diag():Wrong input size.\n");
        return Sparse();
    }
    Sparse M;

    const int n = diagm.nrow();

    //= cholmod_l_allocate_triplet()
      //  sparse_data

    SuiteSparse_long * lin = new SuiteSparse_long[n];
    for (int i = 0; i < n; i++) {
        lin[i] = i;
    }

    cholmod_triplet_struct T;
    T.nrow = n;
    T.ncol = n;
    T.nzmax = n;
    T.nnz = n;
    T.i = (void*)lin;
    T.j = (void*)lin;

    assert(diagm.type() == CHOLMOD_REAL);
    T.x = (void*) (diagm.head());
    T.z = NULL; 

    T.stype = 1; /* -1 could also work. 
                 Describes what parts of the matrix are considered:
         * >0: matrix is square and symmetric.  Entries in the lower triangular
         *     part are transposed and added to the upper triangular part when
         *     the matrix is converted to cholmod_sparse form.
         */

    T.itype = CHOLMOD_LONG; /* CHOLMOD_LONG: i and j are SuiteSparse_long.  Otherwise int */
    T.xtype = CHOLMOD_REAL; /* pattern, real, complex, or zomplex */
    T.dtype = CHOLMOD_DOUBLE; /* x and z are double or float */

    // TODO: free existing memory that is associated with sparse_data

    M.sparse_data = cholmod_l_triplet_to_sparse(&T, 0, M.cm);

    free(lin);

    return M;
}
