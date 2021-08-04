#include "matrix.h"

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

// keep the above same as matrix.cpp

Dense::Dense(int nrow, int ncol, const double init_value, cholmod_common* cm) {
    this->cm = cm;
    int type = INT_MATRIX_DATA_TYPE;
    if (init_value == 0) {
        dense_data = cholmod_l_zeros((SuiteSparse_long)nrow, (SuiteSparse_long)ncol, type, cm);
    }
    else {
        dense_data = cholmod_l_ones((SuiteSparse_long)nrow, (SuiteSparse_long)ncol, type, cm);
        if (init_value == 1) {
            ;
        }
        else {
            RAISE_ERROR("not implemented!");
            // TODO
        }
    }
}

Dense::~Dense() {
    deconstruct();
}

void Dense::deconstruct() {
	cholmod_l_free_dense(&dense_data, cm); // note it asks for cholmod_sparse **
// cholmod_l_finish(&cm);
}

Dense::Dense(const Dense& dense) {
    cm = cm_default();
    *cm = *dense.cm;
    dense_data = cholmod_l_copy_dense(dense.dense_data, cm);
}

Dense& Dense::operator = (const Dense& dense) {
    cholmod_l_free_dense(&dense_data, cm);
	cm = cm_default();
	*cm = *dense.cm;
	dense_data = cholmod_l_copy_dense(dense.dense_data, cm);
}

double& Dense::operator()(int row, int col) const{
    assert(col >= 0 && col < dense_data->ncol);
    assert(row >= 0 && row < dense_data->nrow);
    assert(dense_data->xtype==CHOLMOD_REAL);

    double* Xx = (double*)dense_data->x;
    return Xx[int(row+col*dense_data->d)];
}

Dense Dense::operator + (const Dense& B) const {
    return Dense::add(*this, B);
}

Dense Dense::operator - (const Dense& B) const {
    return Dense::subtract(*this, B);
}

Dense Dense::operator % (const Dense& B) const {
    return Dense::times(*this, B);
}

Dense Dense::operator * (const double& m) const {
    Dense R = Dense::Zeros(this->nrow(), this->ncol());
    Dense::saxy(*this, R, m); // this can be improved. 
    return R;
}

Dense Dense::operator * (const Dense& B) const {
    return mul(B, false); 
}

int Dense::nrow() const {
    return dense_data->nrow;
}

int Dense::ncol() const {
    return dense_data->ncol;
}

int Dense::type() const {
    return dense_data->xtype;
}

void ensure_size(Dense& R, int nrow, int ncol) {
    if ((R.nrow() == nrow) && (R.ncol() == ncol)) { // Lazy copy if size of R does not need to change. 
        ; // do nothing. 
    }
    else {
        cholmod_l_free_dense(&R.dense_data, R.cm);
        int type = INT_MATRIX_DATA_TYPE;
        R.dense_data = cholmod_l_zeros(nrow, ncol, type, R.cm);
    }
}

void Dense::resize(int nrow, int ncol) {
    ensure_size(*this, nrow, ncol);
}

double Dense::norm(int type_of_norm) const {
    // double norm = cholmod_l_norm_dense(dense_data, type_of_norm, cm); /* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
    double norm = my_cholmod_l_norm_dense(dense_data, type_of_norm, cm); /* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
    // my_cholmod_norm_dense can handle dense matrix with more than one column. 
    return norm; 
}

double* Dense::head() const {
    assert(dense_data->xtype == CHOLMOD_REAL);
    return (double *) dense_data->x; 
}

double Dense::trace() const {
    assert(nrow() == ncol());
    double trace = 0;
    for (int i=0; i<nrow(); i++)
    {
        trace += (*this)(i, i);
    }
    return trace;
}

void Dense::slice_assign_value(int* indices, const Dense& B, int axis, int indices_length) {
    
    if (indices_length == -1)
        indices_length = (axis == 0) ? B.nrow() : B.ncol();

    /*
    double* Ax = (double*)this->dense_data->x;
    double* Bx = (double*)B.dense_data->x;
    int nrowA = this->nrow();
    int nrowB = B.nrow();
    */

    if (axis==0) {
        assert(B.nrow() == indices_length);
        for (int i = 0; i < B.nrow(); i++) {
            for (int j = 0; j < B.ncol(); j++) {
                (*this)(indices[i], j) = B(i, j);
            }
        }
    }
    else {
        assert(axis == 1);
        assert(B.ncol() == indices_length);

        for (int j = 0; j < B.ncol(); j++) {
            // (*this)(:, indices[j]) <- B(0, j)
#ifdef DO_NOT_USE_MKL
            for (int i = 0; i < B.nrow(); i++) {
                (*this)(i, indices[j]) = B(i, j);
            }
#else
            cblas_dcopy(B.nrow(), &B(0, j), (MKL_INT)1, &(*this)(0, indices[j]), (MKL_INT)1);
#endif
        }



    }
}

void Dense::slice_assign_value(const DenseInt& indices, const Dense& B, int axis, int indices_length) {
    Dense::slice_assign_value(indices.head(), B, axis, indices_length);
}

void Dense::initTestFor(const Sparse& A) {
    initRHS(dense_data);
}

int Dense::read(std::string filename) {
    FILE* f;
    if ((f = fopen(filename.c_str(), "r")) == NULL)
    {
        printf( //my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
            "unable to open file");
        return -1;
    }

    deconstruct();

    dense_data = cholmod_l_read_dense(f, cm);

    fclose(f);

    assert(dense_data->xtype == INT_MATRIX_DATA_TYPE);
    assert(dense_data->nzmax == dense_data->ncol * dense_data->nrow); 

    return 0;
}

int Dense::write(std::string filename) const {

	assert(dense_data->xtype == INT_MATRIX_DATA_TYPE);
	assert(dense_data->nzmax == dense_data->ncol * dense_data->nrow);

    /*
	int cholmod_write_dense
	(
		FILE * f,		    // file to write to, must already be open 
		cholmod_dense * X,	    // matrix to print 
		const char* comments,    // optional filename of comments to include 
		cholmod_common * Common
	);
    */

	FILE* f;
	if ((f = fopen(filename.c_str(), "wb")) == NULL) {
		printf( //my_handler(CHOLMOD_INVALID, __FILE__, __LINE__,
			"unable to open file");
		return -1;
	}

	cholmod_l_write_dense(f, dense_data, NULL, cm); 
    // Note the comment is the *file* path, not a string! So always set it to NULL is a good idea. 

	fclose(f);

	return 0;
}

Dense Dense::Ones(int nrow, int ncol) {
    Dense ones(nrow, ncol, 1.0); 
    //TODO ones.dense_data
    return ones;
}

Dense Dense::Zeros(int nrow, int ncol) {
    Dense zeros(nrow, ncol, 0.0); // Dense is inited as zero matrix.
    return zeros;
}

void Dense::saxy(const Dense& A, Dense& B, const double alpha) {
	assert(A.nrow() == B.nrow());
	assert(A.ncol() == B.ncol());
	assert(A.type() == B.type());

	double* Ax = (double*)A.dense_data->x;
	double* Bx = (double*)B.dense_data->x;
	// double* Rz = (double*)R.dense_data->z; // for use of complex number. 

#ifdef DO_NOT_USE_MKL

    // naive loop 
    int xtype = A.type();
    if (xtype == CHOLMOD_REAL)
    {
        for (int i = 0; i < B.nrow(); i++)
        {
            for (int j = 0; j < B.ncol(); j++)
            {
                const int p = i + B.nrow() * j;
                Bx[p] = alpha * Ax[p] + Bx[p]; // beta * Bx[p];
            }
        }
    }
    else
    {
        RAISE_ERROR("Not implemented!\n");
    }

#else

    const MKL_INT n = A.nrow() * A.ncol();
    const double a = alpha;
    const double* x = Ax;
    double* y = Bx;
    const MKL_INT incx = 1; // This is absolutely wrong! : sizeof(double);
    const MKL_INT incy = 1; 
	cblas_daxpy(n, a, x, incx, y, incy);

#endif

}

void Dense::add(const Dense& A, const Dense& B, Dense& R) {
    assert(A.nrow() == B.nrow());
    assert(A.ncol() == B.ncol());
    assert(A.type() == B.type());
    
    assert(A.dense_data->d == A.dense_data->nrow); // TODO: this one is optional, remove it later. 
    *R.cm = *A.cm;
    ensure_size(R, A.nrow(), A.ncol());

    double* Rx = (double*)R.dense_data->x;
    double* Ax = (double*)A.dense_data->x;
    double* Bx = (double*)B.dense_data->x;
    // double* Rz = (double*)R.dense_data->z; // for use of complex number. 
    if (false) {
         
        ;
    }
    else {
        // naive loop 
		int xtype = A.type();
		if (xtype == CHOLMOD_REAL)
		{
#ifdef DO_NOT_USE_MKL
            for (int j = 0; j < R.ncol(); j++)
            {

			    for (int i = 0; i < R.nrow(); i++)
			    {
					const int p = i + R.nrow() * j;
					Rx[p] = Ax[p] + Bx[p]; // beta * Bx[p];
				}
			}
#else 
            // vdAddI(R.nrow() * R.ncol(), Ax, (MKL_INT)1, Bx, (MKL_INT)1, Rx, (MKL_INT)1);
            vdAdd(R.nrow() * R.ncol(), Ax, Bx, Rx);
#endif
		}
		else
		{
			RAISE_ERROR("Not implemented!\n");
		}
    }

}

Dense Dense::add(const Dense& A, const Dense& B) {
    Dense R = Dense::Zeros(A.nrow(), A.ncol());
    Dense::add(A, B, R);
    return R;
}

// only broadcasting the second dim of B.
void times_core(const Dense& A, const Dense& B, Dense& R) {
    assert(A.nrow() == B.nrow());
    assert(A.ncol() == B.ncol() || B.ncol()==1); //  only supports broadcasting the second dim of B.
    assert(A.type() == B.type());

    assert(A.dense_data->d == A.dense_data->nrow); // TODO: this one is optional, remove it later. 
    *R.cm = *A.cm;
    ensure_size(R, A.nrow(), A.ncol());

    bool broadcast = A.ncol() != B.ncol();

    double* Rx = (double*)R.dense_data->x;
    double* Ax = (double*)A.dense_data->x;
    double* Bx = (double*)B.dense_data->x;
    // double* Rz = (double*)R.dense_data->z; // for use of complex number. 
    if (false) {

        ;
    }
    else {
        // naive loop 
        int xtype = A.type();
        if (xtype == CHOLMOD_REAL)
        {
            if (!broadcast) 
            {
                for (int j = 0; j < R.ncol(); j++)
                {
#if 0
                    for (int i = 0; i < R.nrow(); i++)
                    {
                    
                        const int p = i + R.nrow() * j;
                        Rx[p] = Ax[p] * Bx[p];
                    }
#else
                    // vdMulI(R.nrow(), &Ax[R.nrow()*j], (MKL_INT)1, &Bx[R.nrow()*j], (MKL_INT)1, &Rx[R.nrow()*j], (MKL_INT)1);
                    vdMul(R.nrow(), &Ax[R.nrow() * j], &Bx[R.nrow() * j], &Rx[R.nrow() * j]);
#endif
                }
            }
            else 
            {
                for (int j = 0; j < R.ncol(); j++)
                {
#if 0
                    for (int i = 0; i < R.nrow(); i++)
                    {
                    
                        const int p = i + R.nrow() * j;
                        Rx[p] = Ax[p] * Bx[i];
                    }
#else
                    // vdMulI(R.nrow(), &Ax[R.nrow() * j], (MKL_INT)1, Bx, (MKL_INT)1, &Rx[R.nrow() * j], (MKL_INT)1);
                    vdMul(R.nrow(), &Ax[R.nrow() * j], Bx, &Rx[R.nrow() * j]);
#endif
                }
            }
        }
        else
        {
            RAISE_ERROR("Not implemented!\n");
        }
    }

}

void Dense::times(const Dense& A, const Dense& B, Dense& R)
{
    assert(A.nrow() == B.nrow());
    if (A.ncol() == B.ncol() || B.ncol() == 1) {
        times_core(A, B, R);
    }
    else {
        assert(A.ncol() == 1);
        times_core(B, A, R);
    }
}

Dense Dense::times(const Dense& A, const Dense& B) {
    Dense R;
    Dense::times(A, B, R);
    return R;
}

void Dense::div(const Dense& A, const Dense& B, Dense& R) {
    assert(A.nrow() == B.nrow());
    assert(A.ncol() == B.ncol());
    assert(A.type() == B.type());

    assert(A.dense_data->d == A.dense_data->nrow); // TODO: this one is optional, remove it later. 
    *R.cm = *A.cm;
    ensure_size(R, A.nrow(), A.ncol());

    double* Rx = (double*)R.dense_data->x;
    double* Ax = (double*)A.dense_data->x;
    double* Bx = (double*)B.dense_data->x;
    // double* Rz = (double*)R.dense_data->z; // for use of complex number. 
    if (false) {

        ;
    }
    else {
        // naive loop 
        int xtype = A.type();
        if (xtype == CHOLMOD_REAL)
        {
#ifdef DO_NOT_USE_MKL
            for (int j = 0; j < R.ncol(); j++)
            {

                for (int i = 0; i < R.nrow(); i++)
                {
                    const int p = i + R.nrow() * j;
                    Rx[p] = Ax[p] / Bx[p]; 
                }
            }
#else 
            // vdDivI(R.nrow() * R.ncol(), Ax, (MKL_INT)1, Bx, (MKL_INT)1, Rx, (MKL_INT)1);
            vdDiv(R.nrow()* R.ncol(), Ax, Bx,  Rx);
#endif
        }
        else
        {
            RAISE_ERROR("Not implemented!\n");
        }
    }

}

Dense Dense::div(const Dense& A, const Dense& B) {
    Dense R = Dense::Zeros(A.nrow(), A.ncol());
    Dense::div(A, B, R);
    return R;
}

void Dense::cross(const Dense& A, const Dense& B, Dense& R) {

    int dim = A.ncol();
    assert(dim == 3);
    assert(B.ncol() == 3);

    int indices[1] = { 0 };

    R.resize(A.nrow(), 3);

    // https://en.wikipedia.org/wiki/Cross_product
    // 1,2 - 2,1
    // 2,0 - 0,2
    // 0,1 - 1,0

    Dense temp = slice_one_col(A, 1) % slice_one_col(B, 2) - slice_one_col(A, 2) % slice_one_col(B, 1);

    // R(:,0) = A(:,1) .* B(:,2) - A(:,2) .* B(:,1);
    indices[0] = 0;  R.slice_assign_value(indices, slice_one_col(A, 1) % slice_one_col(B, 2) - slice_one_col(A, 2) % slice_one_col(B, 1), 1); // axis=1
    indices[0] = 1;  R.slice_assign_value(indices, slice_one_col(A, 2) % slice_one_col(B, 0) - slice_one_col(A, 0) % slice_one_col(B, 2), 1);
    indices[0] = 2;  R.slice_assign_value(indices, slice_one_col(A, 0) % slice_one_col(B, 1) - slice_one_col(A, 1) % slice_one_col(B, 0), 1);

}

Dense Dense::cross(const Dense& A, const Dense& B) {
    Dense R = Dense::Zeros(A.nrow(), A.ncol());
    Dense::cross(A, B, R);
    return R;
}

void Dense::cross2d(const Dense& A, const Dense& B, Dense& R) {

    // essential cross in 3d for planar (2D) vectors (a0,a1,0) and (b0,b1,0)

    int dim = A.ncol();

    assert(dim == 2);
    assert(B.ncol() == 2);

    int indices[1] = { 0 };

    R.resize(A.nrow(), 1);

    indices[0] = 0;  R.slice_assign_value(indices, slice_one_col(A, 0) % slice_one_col(B, 1) - slice_one_col(A, 1) % slice_one_col(B, 0), 1);

}

Dense Dense::cross2d(const Dense& A, const Dense& B) {
    Dense R = Dense::Zeros(A.nrow(), 1);
    Dense::cross2d(A, B, R);
    return R;
}

void Dense::subtract(const Dense& A, const Dense& B, Dense& R) {
    assert(A.nrow() == B.nrow());
    assert(A.ncol() == B.ncol());
    assert(A.type() == B.type());

    assert(A.dense_data->d == A.dense_data->nrow); // TODO: this one is optional, remove it later. 
    *R.cm = *A.cm;
    ensure_size(R, A.nrow(), A.ncol());

    double* Rx = (double*)R.dense_data->x;
    double* Ax = (double*)A.dense_data->x;
    double* Bx = (double*)B.dense_data->x;
    // double* Rz = (double*)R.dense_data->z; // for use of complex number. 
    if (false) {

        ;
    }
    else {
        // naive loop 
        int xtype = A.type();
        if (xtype == CHOLMOD_REAL)
        {
#ifdef DO_NOT_USE_MKL
            for (int j = 0; j < R.ncol(); j++)
            {

                for (int i = 0; i < R.nrow(); i++)
                {
                    const int p = i + R.nrow() * j;
                    Rx[p] =  Ax[p] - Bx[p];
                }
            }
#else 
            // vdSubI(R.nrow() * R.ncol(), Ax, (MKL_INT)1, Bx, (MKL_INT)1, Rx, (MKL_INT)1);
            vdSub(R.nrow()* R.ncol(), Ax, Bx, Rx);
#endif
        }
        else
        {
            RAISE_ERROR("Not implemented!\n");
        }
    }
}

Dense Dense::subtract(const Dense& A, const Dense& B) {
    Dense R = Dense::Zeros(A.nrow(), A.ncol());
    Dense::subtract(A, B, R);
    return R;
}

void Dense::sq(const Dense& A, Dense& R) {

    assert(A.dense_data->d == A.dense_data->nrow); // TODO: this one is optional, remove it later. 
    *R.cm = *A.cm;
    ensure_size(R, A.nrow(), A.ncol());

    double* Rx = (double*)R.dense_data->x;
    double* Ax = (double*)A.dense_data->x;
    // double* Rz = (double*)R.dense_data->z; // for use of complex number. 
    if (false) {

        ;
    }
    else {
        // naive loop 
        int xtype = A.type();
        if (xtype == CHOLMOD_REAL)
        {
            for (int i = 0; i < R.nrow(); i++)
            {
                for (int j = 0; j < R.ncol(); j++)
                {
                    const int p = i + R.nrow() * j;
                    Rx[p] = Ax[p] * Ax[p];
                }
            }
        }
        else
        {
            RAISE_ERROR("Not implemented!\n");
        }
    }

}

Dense Dense::sq(const Dense& A) {
    Dense R = Dense::Zeros(A.nrow(), A.ncol());
    Dense::sq(A, R);
    return R;
}

void Dense::div(const double c, const Dense& A, Dense& R) {

    assert(A.dense_data->d == A.dense_data->nrow); // TODO: this one is optional, remove it later. 
    *R.cm = *A.cm;
    ensure_size(R, A.nrow(), A.ncol());

    double* Rx = (double*)R.dense_data->x;
    double* Ax = (double*)A.dense_data->x;
    // double* Rz = (double*)R.dense_data->z; // for use of complex number. 
    if (false) {

        ;
    }
    else {
        // naive loop 
        int xtype = A.type();
        if (xtype == CHOLMOD_REAL)
        {
            for (int i = 0; i < R.nrow(); i++)
            {
                for (int j = 0; j < R.ncol(); j++)
                {
                    const int p = i + R.nrow() * j;
                    Rx[p] = c / Ax[p];
                }
            }
        }
        else
        {
            RAISE_ERROR("Not implemented!\n");
        }
    }

}

Dense Dense::div(const double c, const Dense& A) {
    Dense R = Dense::Zeros(A.nrow(), A.ncol());
    Dense::div(c, A, R);
    return R;
}


#define GETA(i,j) Ax[(i)+(j)*Arow]
#define GETAT(i,j) Ax[(j)+(i)*Acol]
// GETAT should use Acol, see Acol def later. 
#define GETB(i,j) Bx[(i)+(j)*Brow]
#define GETR(i,j) Rx[(i)+(j)*Arow]

void Dense::mul(const Dense& A, const Dense& B, Dense& R, bool transpose_A) {

	// consider implemented this as: my_cholmod_l_ddmult2()

    int Arow = 0;
    int Acol = 0;
    int Brow = 0;
    int Bcol = 0;

    if (transpose_A) {
		Acol = A.nrow();
        Arow = A.ncol();
    }
    else {
        Arow = A.nrow();
        Acol = A.ncol();
    }

    Brow = B.nrow();
    Bcol = B.ncol();

	assert(Acol == Brow);
    int Row = Acol;

	assert(A.type() == B.type());
	
    *R.cm = *A.cm;
    ensure_size(R, Arow, Bcol);

	double* Rx = (double*)R.dense_data->x;
	double* Ax = (double*)A.dense_data->x;
	double* Bx = (double*)B.dense_data->x;

	int xtype = A.type();
	if (xtype == CHOLMOD_REAL)
	{
		for (int i = 0; i < Arow; i++) {
			for (int j = 0; j < Bcol; j++) {
				double Rij = 0;
				for (int r = 0; r < Row; r++) {
                    if (transpose_A) {
                        Rij += GETAT(i, r) * GETB(r, j);
                    }
                    else {
                        Rij += GETA(i, r) * GETB(r, j);
                    }
				}
				GETR(i, j) = Rij;
			}
		}
	}
	else
	{
		RAISE_ERROR("Not implemented!\n");
	}
}

#undef GETA
#undef GETAT
#undef GETB
#undef GETR

Dense Dense::mul(const Dense& A, const Dense& B, bool transpose_A) {
	Dense R = Dense::Zeros(A.nrow(), B.ncol());
	Dense::mul(A, B, R, transpose_A);
	return R;
}

Dense Dense::mul(const Dense& B, bool transpose_A) const {
    return Dense::mul(*this, B, transpose_A);
}

Dense static_reduce_min(const Dense& A, int axis) {
    
    assert(axis == 0);

    Dense R = Dense::Zeros(1, A.ncol());
    for (int j = 0; j < A.ncol(); j++) {
        double min_vj = 0;
        for (int i = 0; i < A.nrow(); i++) {
            double vij = A(i, j);
            if (i == 0 || vij < min_vj) {
                min_vj = vij;
            }
        }
        R(0, j) = min_vj;
    }
    return R;
}

Dense static_reduce_sum(const Dense& A, int axis) {

    Dense R;

    if (axis == 1) {
        R = Dense::Zeros(A.nrow(), 1);
        for (int j = 0; j < A.ncol(); j++) {
            for (int i = 0; i < A.nrow(); i++) {
                R(i, 0) = R(i, 0) + A(i, j);
            }
        }
    }
    else {
        assert(axis == 0);
        R = Dense::Zeros(1, A.ncol());
        for (int j = 0; j < A.ncol(); j++) {
            for (int i = 0; i < A.nrow(); i++) {
                R(0, j) = R(0, j) + A(i, j);
            }
        }
    }

    return R;
}

#include <math.h>
Dense static_reduce_norm(const Dense& A, int axis) {

    assert(axis == 1);

    Dense R = Dense::Zeros(A.nrow(), 1);
    for (int j = 0; j < A.ncol(); j++) {
        for (int i = 0; i < A.nrow(); i++) {
            R(i, 0) = R(i, 0) + A(i, j) * A(i, j);
        }
    }
    for (int i = 0; i < A.nrow(); i++) {
        R(i, 0) = sqrt(R(i, 0));
    }
    
    return R;
}

Dense static_reduce_sq_sum(const Dense& A, int axis) {

    assert(axis == 1);

    Dense R = Dense::Zeros(A.nrow(), 1);
    for (int j = 0; j < A.ncol(); j++) {
        for (int i = 0; i < A.nrow(); i++) {
            R(i, 0) = R(i, 0) + A(i, j) * A(i, j);
        }
    }

    return R;
}

Dense Dense::reduce_min(int axis) const {
    return static_reduce_min(*this, axis);
}

Dense Dense::reduce_sum(int axis) const {
    return static_reduce_sum(*this, axis);
}

Dense Dense::reduce_norm(int axis) const {
    return static_reduce_norm(*this, axis);
}

Dense Dense::reduce_sq_sum(int axis) const {
    return static_reduce_sq_sum(*this, axis);
}

void Dense::concatenate(const Dense& A, const Dense& B, Dense& R, const int axis) {
    
    if (axis == 0) {
        assert(A.ncol() == B.ncol());
        assert(A.type() == B.type());
        *R.cm = *A.cm;
        ensure_size(R, A.nrow()+B.nrow(), A.ncol());

        double* Rx = (double*)R.dense_data->x;
        double* Ax = (double*)A.dense_data->x;
        double* Bx = (double*)B.dense_data->x;
        // double* Rz = (double*)R.dense_data->z; // for use of complex number. 

        const int nA = A.nrow();
        const int nB = B.nrow();

        int xtype = A.type();
        if (xtype == CHOLMOD_REAL)
        {
#ifdef DO_NOT_USE_MKL
            // old for loop
            for (int j = 0; j < A.ncol(); j++)
            {
                for (int i = 0; i < A.nrow(); i++)
                {
                    const int a = i + nA * j;
                    const int r = i + (nA + nB) * j;
                    Rx[r] = Ax[a];
                }
            }
            for (int j = 0; j < B.ncol(); j++)
            {
                for (int i = 0; i < B.nrow(); i++)
                {
                    const int b = i + nB * j;
                    const int r = nA + i + (nA + nB) * j;
                    Rx[r] = Bx[b];
                }
            }
#else
            // new implementation
            // no performance improvement however...
            for (int j = 0; j < A.ncol(); j++)
            {
                double* aj = &A(0, j);
                double* rj = &R(0, j);
                // memcpy(rj, aj, sizeof(double)*nA);
                cblas_dcopy(nA, aj, (MKL_INT)1, rj, (MKL_INT)1);
                // cblas_dcopy is significantly faster than memcpy
            }
            for (int j = 0; j < B.ncol(); j++)
            {
                double* bj = &B(0, j);
                double* rj = &R(nA, j);
                // memcpy(rj, bj, sizeof(double)*nB);
                cblas_dcopy(nB, bj, (MKL_INT)1, rj, (MKL_INT)1);
            }
#endif
        }
        else
        {
            RAISE_ERROR("Not implemented!\n");
        }
    }
    else {
        assert(axis == 1); 
        RAISE_ERROR("concatenate(): axis==1 not implemented."); 
    }

}

Dense Dense::concatenate(const Dense& A, const Dense& B, const int axis) {
    Dense R(0, 0);
    Dense::concatenate(A, B, R, axis);
    return R;
}

void Dense::print() {
    //cholmod_l_print_dense(dense_data, "Dense Matrix", cm);
    for (int i = 0; i < this->nrow(); i++) {
        for (int j = 0; j < this->ncol(); j++) {
            printf("%f\t", (*this)(i, j));
        }
        printf("\n");
    }

}


DenseInt::DenseInt(int nrow, int ncol, const int init_value): num_rows(nrow), num_cols(ncol) {
    int N = nrow * ncol;
    dense_data = new int[N];
    memset(dense_data, init_value, N*sizeof(int));
}

DenseInt::DenseInt(int* aa, int nrow, int ncol) : num_rows(nrow), num_cols(ncol) {
    int N = nrow * ncol;
    dense_data = new int[N];
    memcpy(dense_data, aa, N * sizeof(int));
}

DenseInt::~DenseInt() {
    delete[] dense_data;
}

void assign_denseint(const DenseInt& src, DenseInt& tar) {
    int N_src = src.nrow() * src.ncol();
    int N_tar = tar.nrow() * tar.ncol();
    if ( N_tar!= N_src) {
        delete[] tar.dense_data;
        tar.dense_data = new int[N_src];
    }
    tar.num_rows = src.nrow();
    tar.num_cols = src.ncol();
    // needs: #include <algorithm>
    // std::copy(std::begin(src.dense_data), std::end(src.dense_data), tar.dense_data);
    memcpy(tar.dense_data, src.dense_data, N_src * (sizeof(int)));
}

DenseInt::DenseInt(const DenseInt& dense) {// copy constructor
    assign_denseint(dense, *this);
}

DenseInt& DenseInt::operator = (const DenseInt& t) { // assignment operator
    assign_denseint(t, *this);
}

int& DenseInt::operator()(int row, int col) const {
    assert(col >= 0 && col < num_cols);
    assert(row >= 0 && row < num_rows);

    return dense_data[row + col * num_rows];
}

int* DenseInt::head() const {
    return dense_data;
}

int DenseInt::nrow() const {
    return num_rows;
}

int DenseInt::ncol() const {
    return num_cols;
}

// funs moved to matrixIO.cpp

void slice_rows(const Dense& A, const DenseInt& I, Dense& R, int icol) 
{
    int ca = A.ncol();
    int ri = I.nrow();

    R = Dense::Zeros(ri, ca);

    for (int i = 0; i < ri; i++) {
        for (int j = 0; j < ca; j++) {
            int ii = I(i, icol);
            R(i, j) = A(ii, j);
        }
    }
}

Dense slice_rows(const Dense& A, const DenseInt& I, int icol) 
{
    Dense R;

    slice_rows(A, I, R, icol);

    return R;
}

void slice_one_col(const Dense& A, Dense& R, int index) 
{
    int ra = A.nrow();

    R = Dense::Zeros(ra, 1);
#ifdef DO_NOT_USE_MKL
    for (int i = 0; i < ra; i++) {
        R(i, 0) = A(i, index);
    }
#else
    cblas_dcopy(ra, &A(0, index), (MKL_INT)1, R.head(), (MKL_INT)1);
#endif
}

Dense slice_one_col(const Dense& A, int index) 
{
    Dense R;

    slice_one_col(A, R, index);

    return R;
}


void symmetric_tensor_rmul(const Sparse& A, const Dense& au, Sparse& B, int dim, int option) {
    // B <- A * symmetric_tensor_assemble(au, dim)

	double* h = au.head();

	int f = au.nrow() / dim;
	assert(au.nrow() % dim == 0);

    my_cholmod_l_sparse_tensor_diag_mul2(A.sparse_data, h, B.sparse_data, option);
}

void sparse_diag_mul(const Sparse& A, const Dense& b, Sparse& B, int start_in_b) {
    
    //printf("TODO: change it back later!\n");
    //return; 

    double* h = b.head();

    my_cholmod_l_sparse_diag_mul2(A.sparse_data, &(h[start_in_b]), B.sparse_data);

}

void sparse_diag_mul_same_pattern(const Sparse** As, const Dense& b, 
    Sparse** Bs, int* starts_in_b, int num) {

    double* h = b.head();

    cholmod_sparse** mA = new cholmod_sparse * [num];
    double** mb = new double*[num];
    cholmod_sparse** mB = new cholmod_sparse * [num];

    for (int i = 0; i < num; i++) {
        mA[i] = (*As[i]).sparse_data;
        mB[i] = (*Bs[i]).sparse_data;
        mb[i] = &(h[starts_in_b[i]]);
    }

    my_cholmod_l_sparse_diag_mul2_same_pattern(
        mA, 
        mb, 
        mB,
        num);

    free(mA);
    free(mB);
    free(mb);
}

// funs moved to matrixIO.cpp
