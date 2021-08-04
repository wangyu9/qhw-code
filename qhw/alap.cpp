
#include "alap.h"

#include "matrix.h"
#include <assert.h>
#include <armadillo>

//#include <string>

// static double delta = 0e-2;
// static double cond_bound = 0.2;

//template <class T>
#define T double

#define REGISTER(name, exp) inline T name(T p1, T p2, T p3, Para para) {T result = (exp); return result;}
#define REGISTER3D(name, exp) inline T name(T p1, T p2, T p3, T p4, T p5, T p6, Para para) {T result = (exp); return result;}

#define ARMA_REGISTER(name, exp) inline arma::vec name(arma::vec p1, arma::vec p2, arma::vec p3, Para para) {arma::vec result = (exp); return result;}
#define ARMA_REGISTER3D(name, exp) inline arma::vec name(arma::vec p1, arma::vec p2, arma::vec p3, arma::vec p4, arma::vec p5, arma::vec p6, Para para) {arma::vec result = (exp); return result;}


#if 0

REGISTER(aA1_2d, p1)
REGISTER(aA2_2d, p2)
REGISTER(aA3_2d, p3)

REGISTER(s_dAdP11_2d, 1)
REGISTER(s_dAdP12_2d, 0)
REGISTER(s_dAdP13_2d, 0)

REGISTER(s_dAdP21_2d, 0)
REGISTER(s_dAdP22_2d, 1)
REGISTER(s_dAdP23_2d, 0)

REGISTER(s_dAdP31_2d, 0)
REGISTER(s_dAdP32_2d, 0)
REGISTER(s_dAdP33_2d, 1)

#else

ARMA_REGISTER(arma_aA1_2d, para.delta + (para.cond_bound * (p2 % p2 + p3 % p3) + 1) / (p1 % p1))
ARMA_REGISTER(arma_aA2_2d, p2 / (p1 % p1))
ARMA_REGISTER(arma_aA3_2d, para.delta + (p2 % p2 + p3 % p3 + para.cond_bound) / (p1 % p1))

ARMA_REGISTER(arma_s_dAdP11_2d, 1.0 / (p1 % p1 % p1) % (para.cond_bound * (p2 % p2 + p3 % p3) + 1.0) * -2.0)
ARMA_REGISTER(arma_s_dAdP12_2d, para.cond_bound * 1.0 / (p1 % p1) % p2 * 2.0)
ARMA_REGISTER(arma_s_dAdP13_2d, para.cond_bound * 1.0 / (p1 % p1) % p3 * 2.0)

ARMA_REGISTER(arma_s_dAdP21_2d, 1.0 / (p1 % p1 % p1) % p2 * -2.0)
ARMA_REGISTER(arma_s_dAdP22_2d, 1.0 / (p1 % p1))
ARMA_REGISTER(arma_s_dAdP23_2d, 0.0 * p1)

ARMA_REGISTER(arma_s_dAdP31_2d, 1.0 / (p1 % p1 % p1) % (para.cond_bound + p2 % p2 + p3 % p3) * -2.0)
ARMA_REGISTER(arma_s_dAdP32_2d, 1.0 / (p1 % p1) % p2 * 2.0)
ARMA_REGISTER(arma_s_dAdP33_2d, 1.0 / (p1 % p1) % p3 * 2.0)

REGISTER(aA1_2d, para.delta + (para.cond_bound * (p2 * p2 + p3 * p3) + 1) / (p1 * p1))
REGISTER(aA2_2d, p2 / (p1 * p1))
REGISTER(aA3_2d, para.delta + (p2 * p2 + p3 * p3 + para.cond_bound) / (p1 * p1))

REGISTER(s_dAdP11_2d, 1.0 / (p1 * p1 * p1) * (para.cond_bound * (p2 * p2 + p3 * p3) + 1.0) * -2.0)
REGISTER(s_dAdP12_2d, para.cond_bound * 1.0 / (p1 * p1) * p2 * 2.0)
REGISTER(s_dAdP13_2d, para.cond_bound * 1.0 / (p1 * p1) * p3 * 2.0)

REGISTER(s_dAdP21_2d, 1.0 / (p1 * p1 * p1) * p2 * -2.0)
REGISTER(s_dAdP22_2d, 1.0 / (p1 * p1))
REGISTER(s_dAdP23_2d, 0.0)

REGISTER(s_dAdP31_2d, 1.0 / (p1 * p1 * p1) * (para.cond_bound + p2 * p2 + p3 * p3) * -2.0)
REGISTER(s_dAdP32_2d, 1.0 / (p1 * p1) * p2 * 2.0)
REGISTER(s_dAdP33_2d, 1.0 / (p1 * p1) * p3 * 2.0)


#endif

#if 0

REGISTER3D(aA1, p1)
REGISTER3D(aA2, p2)
REGISTER3D(aA3, p3)
REGISTER3D(aA4, p4)
REGISTER3D(aA5, p5)
REGISTER3D(aA6, p6)

REGISTER3D(s_dAdP11, 1)
REGISTER3D(s_dAdP12, 0)
REGISTER3D(s_dAdP13, 0)
REGISTER3D(s_dAdP14, 0)
REGISTER3D(s_dAdP15, 0)
REGISTER3D(s_dAdP16, 0)

REGISTER3D(s_dAdP21, 0)
REGISTER3D(s_dAdP22, 1)
REGISTER3D(s_dAdP23, 0)
REGISTER3D(s_dAdP24, 0)
REGISTER3D(s_dAdP25, 0)
REGISTER3D(s_dAdP26, 0)

REGISTER3D(s_dAdP31, 0)
REGISTER3D(s_dAdP32, 0)
REGISTER3D(s_dAdP33, 1)
REGISTER3D(s_dAdP34, 0)
REGISTER3D(s_dAdP35, 0)
REGISTER3D(s_dAdP36, 0)

REGISTER3D(s_dAdP41, 0)
REGISTER3D(s_dAdP42, 0)
REGISTER3D(s_dAdP43, 0)
REGISTER3D(s_dAdP44, 1)
REGISTER3D(s_dAdP45, 0)
REGISTER3D(s_dAdP46, 0)

REGISTER3D(s_dAdP51, 0)
REGISTER3D(s_dAdP52, 0)
REGISTER3D(s_dAdP53, 0)
REGISTER3D(s_dAdP54, 0)
REGISTER3D(s_dAdP55, 1)
REGISTER3D(s_dAdP56, 0)

REGISTER3D(s_dAdP61, 0)
REGISTER3D(s_dAdP62, 0)
REGISTER3D(s_dAdP63, 0)
REGISTER3D(s_dAdP64, 0)
REGISTER3D(s_dAdP65, 0)
REGISTER3D(s_dAdP66, 1)
#else

ARMA_REGISTER3D(arma_aA1, para.delta + (para.cond_bound * (p2 % p2 + p3 % p3 + p4 % p4 + p5 % p5 + p6 % p6) + 1) / (p1 % p1))
ARMA_REGISTER3D(arma_aA2, p2 / (p1 % p1))
ARMA_REGISTER3D(arma_aA3, para.delta + (p2 % p2 + p3 % p3 + para.cond_bound * (p4 % p4 + p5 % p5 + p6 % p6 + 1)) / (p1 % p1))
ARMA_REGISTER3D(arma_aA4, p4 / (p1 % p1))
ARMA_REGISTER3D(arma_aA5, (p2 % p4 + p3 % p5) / (p1 % p1))
ARMA_REGISTER3D(arma_aA6, para.delta + (p4 % p4 + p5 % p5 + p6 % p6 + para.cond_bound * (p2 % p2 + p3 % p3 + 1)) / (p1 % p1))

ARMA_REGISTER3D(arma_s_dAdP11, 1.0 / (p1 % p1 % p1) % (para.cond_bound * (p2 % p2 + p3 % p3 + p4 % p4 + p5 % p5 + p6 % p6) + 1.0) * -2.0)
ARMA_REGISTER3D(arma_s_dAdP12, para.cond_bound * 1.0 / (p1 % p1) % p2 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP13, para.cond_bound * 1.0 / (p1 % p1) % p3 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP14, para.cond_bound * 1.0 / (p1 % p1) % p4 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP15, para.cond_bound * 1.0 / (p1 % p1) % p5 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP16, para.cond_bound * 1.0 / (p1 % p1) % p6 * 2.0)

ARMA_REGISTER3D(arma_s_dAdP21, 1.0 / (p1 % p1 % p1) % p2 * -2.0)
ARMA_REGISTER3D(arma_s_dAdP22, 1.0 / (p1 % p1))
ARMA_REGISTER3D(arma_s_dAdP23, 0.0 * p2)
ARMA_REGISTER3D(arma_s_dAdP24, 0.0 * p2)
ARMA_REGISTER3D(arma_s_dAdP25, 0.0 * p2)
ARMA_REGISTER3D(arma_s_dAdP26, 0.0 * p2)

ARMA_REGISTER3D(arma_s_dAdP31, 1.0 / (p1 % p1 % p1) % (para.cond_bound * (p4 % p4 + p5 % p5 + p6 % p6 + 1.0) + p2 % p2 + p3 % p3) * -2.0)
ARMA_REGISTER3D(arma_s_dAdP32, 1.0 / (p1 % p1) % p2 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP33, 1.0 / (p1 % p1) % p3 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP34, para.cond_bound * 1.0 / (p1 % p1) % p4 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP35, para.cond_bound * 1.0 / (p1 % p1) % p5 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP36, para.cond_bound * 1.0 / (p1 % p1) % p6 * 2.0)

ARMA_REGISTER3D(arma_s_dAdP41, 1.0 / (p1 % p1 % p1) % p4 * -2.0)
ARMA_REGISTER3D(arma_s_dAdP42, 0.0 * p2)
ARMA_REGISTER3D(arma_s_dAdP43, 0.0 * p2)
ARMA_REGISTER3D(arma_s_dAdP44, 1.0 / (p1 % p1))
ARMA_REGISTER3D(arma_s_dAdP45, 0.0 * p2)
ARMA_REGISTER3D(arma_s_dAdP46, 0.0 * p2)

ARMA_REGISTER3D(arma_s_dAdP51, 1.0 / (p1 % p1 % p1) % (p2 % p4 + p3 % p5) * -2.0)
ARMA_REGISTER3D(arma_s_dAdP52, 1.0 / (p1 % p1) % p4)
ARMA_REGISTER3D(arma_s_dAdP53, 1.0 / (p1 % p1) % p5)
ARMA_REGISTER3D(arma_s_dAdP54, 1.0 / (p1 % p1) % p2)
ARMA_REGISTER3D(arma_s_dAdP55, 1.0 / (p1 % p1) % p3)
ARMA_REGISTER3D(arma_s_dAdP56, 0.0 * p2)

ARMA_REGISTER3D(arma_s_dAdP61, 1.0 / (p1 % p1 % p1) % (para.cond_bound * (p2 % p2 + p3 % p3 + 1.0) + p4 % p4 + p5 % p5 + p6 % p6) * -2.0)
ARMA_REGISTER3D(arma_s_dAdP62, para.cond_bound * 1.0 / (p1 % p1) % p2 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP63, para.cond_bound * 1.0 / (p1 % p1) % p3 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP64, 1.0 / (p1 % p1) % p4 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP65, 1.0 / (p1 % p1) % p5 * 2.0)
ARMA_REGISTER3D(arma_s_dAdP66, 1.0 / (p1 % p1) % p6 * 2.0)

REGISTER3D(aA1, para.delta + (para.cond_bound * (p2 * p2 + p3 * p3 + p4 * p4 + p5 * p5 + p6 * p6) + 1) / (p1 * p1) )
REGISTER3D(aA2, p2 / (p1 * p1) )
REGISTER3D(aA3, para.delta + (p2 * p2 + p3 * p3 + para.cond_bound * (p4 * p4 + p5 * p5 + p6 * p6 + 1)) / (p1 * p1) )
REGISTER3D(aA4, p4 / (p1 * p1) )
REGISTER3D(aA5, (p2* p4 + p3 * p5) / (p1 * p1) )
REGISTER3D(aA6, para.delta + (p4 * p4 + p5 * p5 + p6 * p6 + para.cond_bound * (p2 * p2 + p3 * p3 + 1)) / (p1 * p1) )

REGISTER3D(s_dAdP11, 1.0 / (p1 * p1 * p1) * (para.cond_bound * (p2 * p2 + p3 * p3 + p4 * p4 + p5 * p5 + p6 * p6) + 1.0) * -2.0 )
REGISTER3D(s_dAdP12, para.cond_bound * 1.0 / (p1 * p1) * p2 * 2.0 )
REGISTER3D(s_dAdP13, para.cond_bound * 1.0 / (p1 * p1) * p3 * 2.0 )
REGISTER3D(s_dAdP14, para.cond_bound * 1.0 / (p1 * p1) * p4 * 2.0 )
REGISTER3D(s_dAdP15, para.cond_bound * 1.0 / (p1 * p1) * p5 * 2.0 )
REGISTER3D(s_dAdP16, para.cond_bound * 1.0 / (p1 * p1) * p6 * 2.0 )

REGISTER3D(s_dAdP21, 1.0 / (p1 * p1 * p1) * p2 * -2.0 )
REGISTER3D(s_dAdP22, 1.0 / (p1 * p1) )
REGISTER3D(s_dAdP23, 0.0 )
REGISTER3D(s_dAdP24, 0.0 )
REGISTER3D(s_dAdP25, 0.0 )
REGISTER3D(s_dAdP26, 0.0 )

REGISTER3D(s_dAdP31, 1.0 / (p1 * p1 * p1) * (para.cond_bound * (p4 * p4 + p5 * p5 + p6 * p6 + 1.0) + p2 * p2 + p3 * p3) * -2.0 )
REGISTER3D(s_dAdP32, 1.0 / (p1 * p1) * p2 * 2.0 )
REGISTER3D(s_dAdP33, 1.0 / (p1 * p1) * p3 * 2.0 )
REGISTER3D(s_dAdP34, para.cond_bound * 1.0 / (p1 * p1) * p4 * 2.0 )
REGISTER3D(s_dAdP35, para.cond_bound * 1.0 / (p1 * p1) * p5 * 2.0 )
REGISTER3D(s_dAdP36, para.cond_bound * 1.0 / (p1 * p1) * p6 * 2.0 )

REGISTER3D(s_dAdP41, 1.0 / (p1 * p1 * p1) * p4 * -2.0 )
REGISTER3D(s_dAdP42, 0.0 )
REGISTER3D(s_dAdP43, 0.0 )
REGISTER3D(s_dAdP44, 1.0 / (p1 * p1) )
REGISTER3D(s_dAdP45, 0.0 )
REGISTER3D(s_dAdP46, 0.0 )

REGISTER3D(s_dAdP51, 1.0 / (p1 * p1 * p1) * (p2 * p4 + p3 * p5) * -2.0 )
REGISTER3D(s_dAdP52, 1.0 / (p1 * p1) * p4 )
REGISTER3D(s_dAdP53, 1.0 / (p1 * p1) * p5 )
REGISTER3D(s_dAdP54, 1.0 / (p1 * p1) * p2 )
REGISTER3D(s_dAdP55, 1.0 / (p1 * p1) * p3 )
REGISTER3D(s_dAdP56, 0.0 )

REGISTER3D(s_dAdP61, 1.0 / (p1 * p1 * p1) * (para.cond_bound * (p2 * p2 + p3 * p3 + 1.0) + p4 * p4 + p5 * p5 + p6 * p6) * -2.0 )
REGISTER3D(s_dAdP62, para.cond_bound * 1.0 / (p1 * p1) * p2 * 2.0 )
REGISTER3D(s_dAdP63, para.cond_bound * 1.0 / (p1 * p1) * p3 * 2.0 )
REGISTER3D(s_dAdP64, 1.0 / (p1 * p1) * p4 * 2.0 )
REGISTER3D(s_dAdP65, 1.0 / (p1 * p1) * p5 * 2.0 )
REGISTER3D(s_dAdP66, 1.0 / (p1 * p1) * p6 * 2.0 )
#endif

// 

/*
inline T aA2(T p1, T p2, T p3) {
	T result = p2 / (p1*p1);
	return (result);
}
*/

#undef T

// au holds in the order of [a00,a01,a11;]; 
#if 0
// equivalent way of writing. 
#define DAT(i,j) (at((i)+(f)*j,0))
#define DAU(i,j) (au((i)+(f)*j,0))
// TODO complete other defs. 
#else
// #define DAT(i,j) (((double*)at.dense_data->x)[(i)+(f)*j])
#define DAT(i,j) (h_at[(i)+(f)*j])
#define DAU(i,j) (((double*)au.dense_data->x)[(i)+(f)*j])
// #define GAT(i,j) (((double*)g_at.dense_data->x)[(i)+(f)*j])
#define GAT(i,j) (h_gat[(i)+(f)*j])
#define GAU(i,j) (((double*)g_au.dense_data->x)[(i)+(f)*j])
#endif

#define INPUTS DAT(i, 0), DAT(i, 1), DAT(i, 2)
#define INPUTS3D DAT(i, 0), DAT(i, 1), DAT(i, 2), DAT(i, 3), DAT(i, 4), DAT(i, 5)

// #define TDAT(i,j) (((double*)at.dense_data->x)[(i)*3+(j)])
// #define TDAU(i,j) (((double*)au.dense_data->x)[(i)*3+(j)])


void s_at2au_fast(arma::vec& at, arma::vec& au, const arma::vec& FA, const int dim, const Para para) {

	// memptr not allowing at to be const. 

	const int f = FA.n_rows; // no need to divide by dim!

	const arma::vec at1(at.memptr(), f, false, true);
	const arma::vec at2(at.memptr() + f, f, false, true);
	const arma::vec at3(at.memptr() + 2 * f, f, false, true);

	arma::vec au1(au.memptr(), f, false, true);
	arma::vec au2(au.memptr() + f, f, false, true);
	arma::vec au3(au.memptr() + 2 * f, f, false, true);

	if (dim == 2) {
		au1 = arma_aA1_2d(at1, at2, at3, para) % FA;
		au2 = arma_aA2_2d(at1, at2, at3, para) % FA;
		au3 = arma_aA3_2d(at1, at2, at3, para) % FA;
	}
	else
	{
		assert(dim == 3);

		const arma::vec at4(at.memptr() + 3 * f, f, false, true);
		const arma::vec at5(at.memptr() + 4 * f, f, false, true);
		const arma::vec at6(at.memptr() + 5 * f, f, false, true);

		arma::vec au4(au.memptr() + 3 * f, f, false, true);
		arma::vec au5(au.memptr() + 4 * f, f, false, true);
		arma::vec au6(au.memptr() + 5 * f, f, false, true);

		au1 = arma_aA1(at1, at2, at3, at4, at5, at6, para) % FA;
		au2 = arma_aA2(at1, at2, at3, at4, at5, at6, para) % FA;
		au3 = arma_aA3(at1, at2, at3, at4, at5, at6, para) % FA;
		au4 = arma_aA4(at1, at2, at3, at4, at5, at6, para) % FA;
		au5 = arma_aA5(at1, at2, at3, at4, at5, at6, para) % FA;
		au6 = arma_aA6(at1, at2, at3, at4, at5, at6, para) % FA;
	}

}

void s_at2au_fast(arma::vec& at, Dense& au, const Dense& FA, const int dim, const Para para) {

	// http://arma.sourceforge.net/docs.html#Col

	// from writable auxiliary (external) memory

	arma::vec arma_FA(FA.head(), FA.nrow(), false, true);

	arma::vec arma_au(au.head(), au.nrow(), false, true);

	s_at2au_fast(at, arma_au, arma_FA, dim, para);
}

// void s_at2au(const Dense& at, Dense& au, const Dense& FA, const int dim = 2) {
void s_at2au(const double * h_at, Dense& au, const Dense& FA, const int dim, const Para para) {
	const int f = FA.nrow(); // no need to divide by dim!
	const MKL_INT N = f;
	const MKL_INT incX = 1;
	const MKL_INT incY = 1;
	assert(sizeof(MKL_INT) == sizeof(long));
	const double alpha = 1;
	double* fa = FA.head();

	if (dim == 2) {

		for (int i = 0; i < f; i++) {

			DAU(i, 0) = aA1_2d(INPUTS, para) * FA(i, 0);
			DAU(i, 1) = aA2_2d(INPUTS, para) * FA(i, 0);
			DAU(i, 2) = aA3_2d(INPUTS, para) * FA(i, 0);
		}
	}
	else {
		assert(dim == 3);

		for (int i = 0; i < f; i++) {

			DAU(i, 0) = aA1(INPUTS3D, para) * FA(i, 0);
			DAU(i, 1) = aA2(INPUTS3D, para) * FA(i, 0);
			DAU(i, 2) = aA3(INPUTS3D, para) * FA(i, 0);
			DAU(i, 3) = aA4(INPUTS3D, para) * FA(i, 0);
			DAU(i, 4) = aA5(INPUTS3D, para) * FA(i, 0);
			DAU(i, 5) = aA6(INPUTS3D, para) * FA(i, 0);
		}
	}
}

void s_at2au_1(const double* h_at, Dense& au, const Dense& FA, const int dim, const Para para) {
	const int f = FA.nrow(); // no need to divide by dim!
	const MKL_INT N = f;
	const MKL_INT incX = 1;
	const MKL_INT incY = 1;
	assert(sizeof(MKL_INT) == sizeof(long));
	const double alpha = 1;
	double* fa = FA.head();

	if (dim == 2) {
		for (int j = 0; j < 3; j++) {
			const double* X = &DAT(0, j);
			double* Y = &DAU(0, j);
			if (false) {

				// This is wrong! Since y may not be 0 to start with. 
				cblas_daxpy(N, alpha, X, incX, Y, incY); // y := a*x + y
			}
			else {
				// cblas_dcopy(N, X, incX, Y, incY);
				vdMul(N, X, fa, Y);
			}
		}
	}
	else {
		assert(dim == 3);
		for (int j = 0; j < 6; j++) {
			const double* X = &DAT(0, j);
			double* Y = &DAU(0, j);
			// cblas_dcopy(N, X, incX, Y, incY);
			vdMul(N, X, fa, Y);
		}
	}
}

void s_at2au(const Dense& at, Dense& au, const Dense& FA, const int dim, const Para para) {
	s_at2au((double*)at.dense_data->x, au, FA, dim, para);
}

#ifdef USE_EIGEN
void s_at2au(const Eigen::VectorXd& at, Dense& au, const Dense& FA, const int dim, const Para para) {
	s_at2au( &at.coeffRef(0), au, FA, dim, para);
}
#endif

void s_at2au(const arma::vec& at, Dense& au, const Dense& FA, const int dim, const Para para) {
	s_at2au(at.colptr(0), au, FA, dim, para);
}

void s_pdapdt_lmul_fast(arma::vec& at, arma::vec& g_au, arma::vec& g_at, const arma::vec& FA, const int dim, const Para para) {

	const int f = FA.n_rows; // no need to divide by dim!

	const arma::vec at1(at.memptr(), f, false, true);
	const arma::vec at2(at.memptr() + f, f, false, true);
	const arma::vec at3(at.memptr() + 2 * f, f, false, true);

	arma::vec gau1(g_au.memptr(), f, false, true);
	arma::vec gau2(g_au.memptr() + f, f, false, true);
	arma::vec gau3(g_au.memptr() + 2 * f, f, false, true);

	arma::vec gat1(g_at.memptr(), f, false, true);
	arma::vec gat2(g_at.memptr() + f, f, false, true);
	arma::vec gat3(g_at.memptr() + 2 * f, f, false, true);

	if (dim == 2) {

		gat1 = (gau1 % arma_s_dAdP11_2d(at1, at2, at3, para) + gau2 % arma_s_dAdP21_2d(at1, at2, at3, para) + gau3 % arma_s_dAdP31_2d(at1, at2, at3, para)) % FA;
		gat2 = (gau1 % arma_s_dAdP12_2d(at1, at2, at3, para) + gau2 % arma_s_dAdP22_2d(at1, at2, at3, para) + gau3 % arma_s_dAdP32_2d(at1, at2, at3, para)) % FA;
		gat3 = (gau1 % arma_s_dAdP13_2d(at1, at2, at3, para) + gau2 % arma_s_dAdP23_2d(at1, at2, at3, para) + gau3 % arma_s_dAdP33_2d(at1, at2, at3, para)) % FA;

	}
	else {
		assert(dim == 3);

		const arma::vec at4(at.memptr() + 3 * f, f, false, true);
		const arma::vec at5(at.memptr() + 4 * f, f, false, true);
		const arma::vec at6(at.memptr() + 5 * f, f, false, true);

		arma::vec gau4(g_au.memptr() + 3 * f, f, false, true);
		arma::vec gau5(g_au.memptr() + 4 * f, f, false, true);
		arma::vec gau6(g_au.memptr() + 5 * f, f, false, true);

		arma::vec gat4(g_at.memptr() + 3 * f, f, false, true);
		arma::vec gat5(g_at.memptr() + 4 * f, f, false, true);
		arma::vec gat6(g_at.memptr() + 5 * f, f, false, true);

		gat1 =	( gau1 % arma_s_dAdP11(at1, at2, at3, at4, at5, at6, para) + gau2 % arma_s_dAdP21(at1, at2, at3, at4, at5, at6, para)
				+ gau3 % arma_s_dAdP31(at1, at2, at3, at4, at5, at6, para) + gau4 % arma_s_dAdP41(at1, at2, at3, at4, at5, at6, para)
				+ gau5 % arma_s_dAdP51(at1, at2, at3, at4, at5, at6, para) + gau6 % arma_s_dAdP61(at1, at2, at3, at4, at5, at6, para)) % FA;

		gat2 =	( gau1 % arma_s_dAdP12(at1, at2, at3, at4, at5, at6, para) + gau2 % arma_s_dAdP22(at1, at2, at3, at4, at5, at6, para)
				+ gau3 % arma_s_dAdP32(at1, at2, at3, at4, at5, at6, para) + gau4 % arma_s_dAdP42(at1, at2, at3, at4, at5, at6, para)
				+ gau5 % arma_s_dAdP52(at1, at2, at3, at4, at5, at6, para) + gau6 % arma_s_dAdP62(at1, at2, at3, at4, at5, at6, para)) % FA;

		gat3 =	( gau1 % arma_s_dAdP13(at1, at2, at3, at4, at5, at6, para) + gau2 % arma_s_dAdP23(at1, at2, at3, at4, at5, at6, para)
				+ gau3 % arma_s_dAdP33(at1, at2, at3, at4, at5, at6, para) + gau4 % arma_s_dAdP43(at1, at2, at3, at4, at5, at6, para)
				+ gau5 % arma_s_dAdP53(at1, at2, at3, at4, at5, at6, para) + gau6 % arma_s_dAdP63(at1, at2, at3, at4, at5, at6, para)) % FA;

		gat4 =	( gau1 % arma_s_dAdP14(at1, at2, at3, at4, at5, at6, para) + gau2 % arma_s_dAdP24(at1, at2, at3, at4, at5, at6, para)
				+ gau3 % arma_s_dAdP34(at1, at2, at3, at4, at5, at6, para) + gau4 % arma_s_dAdP44(at1, at2, at3, at4, at5, at6, para)
				+ gau5 % arma_s_dAdP54(at1, at2, at3, at4, at5, at6, para) + gau6 % arma_s_dAdP64(at1, at2, at3, at4, at5, at6, para)) % FA;

		gat5 =	( gau1 % arma_s_dAdP15(at1, at2, at3, at4, at5, at6, para) + gau2 % arma_s_dAdP25(at1, at2, at3, at4, at5, at6, para)
				+ gau3 % arma_s_dAdP35(at1, at2, at3, at4, at5, at6, para) + gau4 % arma_s_dAdP45(at1, at2, at3, at4, at5, at6, para)
				+ gau5 % arma_s_dAdP55(at1, at2, at3, at4, at5, at6, para) + gau6 % arma_s_dAdP65(at1, at2, at3, at4, at5, at6, para)) % FA;

		gat6 =	( gau1 % arma_s_dAdP16(at1, at2, at3, at4, at5, at6, para) + gau2 % arma_s_dAdP26(at1, at2, at3, at4, at5, at6, para)
				+ gau3 % arma_s_dAdP36(at1, at2, at3, at4, at5, at6, para) + gau4 % arma_s_dAdP46(at1, at2, at3, at4, at5, at6, para)
				+ gau5 % arma_s_dAdP56(at1, at2, at3, at4, at5, at6, para) + gau6 % arma_s_dAdP66(at1, at2, at3, at4, at5, at6, para)) % FA;

	}
}

void s_pdapdt_lmul_fast(arma::vec& at, Dense& g_au, arma::vec& g_at, const Dense& FA, const int dim, const Para para) {
	arma::vec arma_FA(FA.head(), FA.nrow(), false, true);

	arma::vec arma_gau(g_au.head(), g_au.nrow(), false, true);

	s_pdapdt_lmul_fast(at, arma_gau, g_at, arma_FA, dim, para);
}

//
// void s_at2au_0(const Dense& at, Dense& au, const Dense& FA, const int dim = 2) {
void s_at2au_0(const double* h_at, Dense& au, const Dense& FA, const int dim, const Para para) {

	const int f = FA.nrow(); // no need to divide by dim!
	if (dim == 2) {
#if 1
		for (int i = 0; i < f; i++) {
			//  at((i)+(f)*j,0); 

			DAU(i, 0) = aA1_2d(INPUTS, para);
			DAU(i, 1) = aA2_2d(INPUTS, para);
			DAU(i, 2) = aA3_2d(INPUTS, para);
		}
#else
		// This is wrong code, just for testing performance purpose. 
		for (int i = 0; i < f/2; i++) {
			TDAU(i, 0) = TDAT(i, 0);
			TDAU(i, 1) = TDAT(i, 1);
			TDAU(i, 2) = TDAT(i, 2);
			TDAU(i, 3) = TDAT(i, 3);
		}
#endif	
	}
	else {
		assert(dim == 3);
		assert(false && "Not implemented!\n");
	}
}

// void s_pdapdt_lmul(const Dense& at, const Dense& g_au, Dense& g_at, const Dense& FA, const int dim = 2) {
void s_pdapdt_lmul(const double* h_at, const Dense& g_au, double* h_gat, const Dense& FA, const int dim, const Para para) {
	// assert(at.nrow() == g_at.nrow());
	// assert(g_au.nrow() == g_at.nrow());
	const int f = FA.nrow();
	if (dim == 2) {
		for (int i = 0; i < f; i++) {
			GAT(i, 0) = GAU(i, 0) * s_dAdP11_2d(INPUTS, para) + GAU(i, 1) * s_dAdP21_2d(INPUTS, para) + GAU(i, 2) * s_dAdP31_2d(INPUTS, para);
			GAT(i, 1) = GAU(i, 0) * s_dAdP12_2d(INPUTS, para) + GAU(i, 1) * s_dAdP22_2d(INPUTS, para) + GAU(i, 2) * s_dAdP32_2d(INPUTS, para);
			GAT(i, 2) = GAU(i, 0) * s_dAdP13_2d(INPUTS, para) + GAU(i, 1) * s_dAdP23_2d(INPUTS, para) + GAU(i, 2) * s_dAdP33_2d(INPUTS, para);
			GAT(i, 0) *= FA(i, 0);
			GAT(i, 1) *= FA(i, 0);
			GAT(i, 2) *= FA(i, 0);
		}
	}
	else {
		assert(dim == 3);
		for (int i = 0; i < f; i++) {
			GAT(i, 0) =	GAU(i, 0) * s_dAdP11(INPUTS3D, para) + GAU(i, 1) * s_dAdP21(INPUTS3D, para) + GAU(i, 2) * s_dAdP31(INPUTS3D, para)
					+	GAU(i, 3) * s_dAdP41(INPUTS3D, para) + GAU(i, 4) * s_dAdP51(INPUTS3D, para) + GAU(i, 5) * s_dAdP61(INPUTS3D, para);

			GAT(i, 1) = GAU(i, 0) * s_dAdP12(INPUTS3D, para) + GAU(i, 1) * s_dAdP22(INPUTS3D, para) + GAU(i, 2) * s_dAdP32(INPUTS3D, para)
					+	GAU(i, 3) * s_dAdP42(INPUTS3D, para) + GAU(i, 4) * s_dAdP52(INPUTS3D, para) + GAU(i, 5) * s_dAdP62(INPUTS3D, para);

			GAT(i, 2) = GAU(i, 0) * s_dAdP13(INPUTS3D, para) + GAU(i, 1) * s_dAdP23(INPUTS3D, para) + GAU(i, 2) * s_dAdP33(INPUTS3D, para)
					+	GAU(i, 3) * s_dAdP43(INPUTS3D, para) + GAU(i, 4) * s_dAdP53(INPUTS3D, para) + GAU(i, 5) * s_dAdP63(INPUTS3D, para);

			GAT(i, 3) = GAU(i, 0) * s_dAdP14(INPUTS3D, para) + GAU(i, 1) * s_dAdP24(INPUTS3D, para) + GAU(i, 2) * s_dAdP34(INPUTS3D, para)
					+	GAU(i, 3) * s_dAdP44(INPUTS3D, para) + GAU(i, 4) * s_dAdP54(INPUTS3D, para) + GAU(i, 5) * s_dAdP64(INPUTS3D, para);

			GAT(i, 4) = GAU(i, 0) * s_dAdP15(INPUTS3D, para) + GAU(i, 1) * s_dAdP25(INPUTS3D, para) + GAU(i, 2) * s_dAdP35(INPUTS3D, para)
					+	GAU(i, 3) * s_dAdP45(INPUTS3D, para) + GAU(i, 4) * s_dAdP55(INPUTS3D, para) + GAU(i, 5) * s_dAdP65(INPUTS3D, para);

			GAT(i, 5) = GAU(i, 0) * s_dAdP16(INPUTS3D, para) + GAU(i, 1) * s_dAdP26(INPUTS3D, para) + GAU(i, 2) * s_dAdP36(INPUTS3D, para)
					+	GAU(i, 3) * s_dAdP46(INPUTS3D, para) + GAU(i, 4) * s_dAdP56(INPUTS3D, para) + GAU(i, 5) * s_dAdP66(INPUTS3D, para);

			GAT(i, 0) *= FA(i, 0);
			GAT(i, 1) *= FA(i, 0);
			GAT(i, 2) *= FA(i, 0);
			GAT(i, 3) *= FA(i, 0);
			GAT(i, 4) *= FA(i, 0);
			GAT(i, 5) *= FA(i, 0);
		}
	}
}

void s_pdapdt_lmul(const Dense& at, const Dense& g_au, Dense& g_at, const Dense& FA, const int dim, const Para para) {
	s_pdapdt_lmul((double*)at.dense_data->x, g_au, (double*)g_at.dense_data->x, FA, dim, para);
}

#ifdef USE_EIGEN
void s_pdapdt_lmul(const Eigen::VectorXd& at, const Dense& g_au, Eigen::VectorXd& g_at, const Dense& FA, const int dim, const Para para) {
	s_pdapdt_lmul( &at.coeffRef(0), g_au, &g_at.coeffRef(0), FA, dim, para);
}
#endif

void s_pdapdt_lmul(const arma::vec& at, const Dense& g_au, arma::vec& g_at, const Dense& FA, const int dim, const Para para) {
#if 0
	Dense dense_g_at = Dense::Zeros(g_au.nrow(), 1);
	s_pdapdt_lmul(at.colptr(0), g_au, (double*)dense_g_at.dense_data->x, FA, dim); // g_at.memptr() or colptr(0) is not writable.  
	g_at = dense_array_to_arma_vec(dense_g_at); 
#else
	s_pdapdt_lmul(at.colptr(0), g_au, g_at.colptr(0), FA, dim, para);
#endif
}

void symmetric_tensor_assemble_indices(int*& I, int*& J, int f, int dim) {
	const int mcdim = (dim == 2) ? 3 : 6;
	I = new int[f * mcdim]; // mcdim: 3 or 6
	J = new int[f * mcdim];
	{
		// this works for mcdim==3 only. 
		// So I, J should not be used for mcdim==6.
		// better than doing all three in one loop: 
		for (int i = 0; i < f; i++) {
			// A00
			I[i] = i;
			J[i] = i;
		}
		for (int i = 0; i < f; i++) {
			// A10
			I[i + f] = i + f;
			J[i + f] = i;
		}
		for (int i = 0; i < f; i++) {
			// A11
			I[i + f * 2] = i + f;
			J[i + f * 2] = i + f;
		}
	}
	if (dim == 3) {
		for (int i = 0; i < f; i++) {
			// A20
			I[i + f * 3] = i + f * 2;
			J[i + f * 3] = i;
		}
		for (int i = 0; i < f; i++) {
			// A21
			I[i + f * 4] = i + f * 2;
			J[i + f * 4] = i + f;
		}
		for (int i = 0; i < f; i++) {
			// A22
			I[i + f * 5] = i + f * 2;
			J[i + f * 5] = i + f * 2;
		}
	}
}

Sparse symmetric_tensor_assemble(const Dense& au, const int dim, int * I, int * J) { // Sparse& AA
	int f;
	const int mcdim = (dim == 2) ? 3 : 6;
	if (dim == 2) {
		f = au.nrow() / 3;
		//if (AA.nrow() != 2 * f) { // Setup the sparsity pattern of A if it is for the first time. 
		// Always do this since it is not clear how matrix is reordered.

		//for (int i = 0; i < f; i++) {
		//    DAU(i, 0); // A00
		//    DAU(i, 1); // A01=A10
		//    DAU(i, 2); // A11
		//}
	}
	else {
		assert(dim == 3);
		f = au.nrow() / 6;
	}
		  
	double* V = au.head();
	const int stype = 1; // 1: symmetric. The lower triangular part is transposed and added to the upper
	Sparse AA = Sparse(I, J, V, dim * f, dim * f, mcdim * f, stype);
	return AA;
	
}


#define DDB(i,j) ((double*)b.dense_data->x)[(i)+(f)*j]
#define DDC(i,j) ((double*)c.dense_data->x)[(i)+(f)*j]
#define DDP(i,j) ((double*)p.dense_data->x)[(i)+(f)*j]

// Output: p
void symmetric_tensor_span_dot(const Dense& b, const Dense& c, Dense& p, int dim) {

	assert(b.nrow() == c.nrow());
	const int f = b.nrow() / dim;
	assert(p.nrow() == 3 * f);

	// Keep the code since here across symmetric_tensor_span_dot

	if (dim == 2) {
		for (int i = 0; i < f; i++) {
			DDP(i, 0) = DDB(i, 0) * DDC(i, 0);
			DDP(i, 1) = DDB(i, 1) * DDC(i, 0) + DDB(i, 0) * DDC(i, 1);
			DDP(i, 2) = DDB(i, 1) * DDC(i, 1);
		}
	}
	else {
		assert(dim == 3);
		assert(false); // not implemented. 
	}
}

#undef DDB
#undef DDC
#undef DDP 
#define DDB(i,j) (b)[(i)+(f)*j]
#define DDC(i,j) (c)[(i)+(f)*j]
#define DDP(i,j) (p)[(i)+(f)*j]

void symmetric_tensor_span_dot(const double* b, const double* c, double* p, int f, int dim) {

	// Keep the code since here across symmetric_tensor_span_dot

	if (dim == 2) {
		for (int i = 0; i < f; i++) {
			DDP(i, 0) = DDB(i, 0) * DDC(i, 0);
			DDP(i, 1) = DDB(i, 1) * DDC(i, 0) + DDB(i, 0) * DDC(i, 1);
			DDP(i, 2) = DDB(i, 1) * DDC(i, 1);
		}
	}
	else {
		assert(dim == 3);
		for (int i = 0; i < f; i++) {
			DDP(i, 0) = DDB(i, 0) * DDC(i, 0);
			DDP(i, 1) = DDB(i, 1) * DDC(i, 0) + DDB(i, 0) * DDC(i, 1);
			DDP(i, 2) = DDB(i, 1) * DDC(i, 1);
			DDP(i, 3) = DDB(i, 2) * DDC(i, 0) + DDB(i, 0) * DDC(i, 2);
			DDP(i, 4) = DDB(i, 2) * DDC(i, 1) + DDB(i, 1) * DDC(i, 2);
			DDP(i, 5) = DDB(i, 2) * DDC(i, 2);
		}
	}
}

#undef DDB
#undef DDC
#undef DDP 

#undef DAT
#undef DAU
#undef INPUTS

