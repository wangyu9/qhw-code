#pragma once

#include "matrix.h"
#include <assert.h>
#include <armadillo>

#include "mkl.h"
#include "mkl_vml.h"

#ifdef USE_EIGEN
#include <Eigen/Core>
#endif

class Para {
public:
	double delta = 0e-2;
	double cond_bound = 0.2;
};

void s_at2au_fast(arma::vec& at, arma::vec& au, const arma::vec& FA, const int dim, const Para para);

void s_at2au_fast(arma::vec& at, Dense& au, const Dense& FA, const int dim, const Para para);

// void s_at2au(const Dense& at, Dense& au, const Dense& FA, const int dim = 2) {
void s_at2au(const double* h_at, Dense& au, const Dense& FA, const int dim, const Para para);

void s_at2au_1(const double* h_at, Dense& au, const Dense& FA, const int dim, const Para para);

void s_at2au(const Dense& at, Dense& au, const Dense& FA, const int dim, const Para para);

#ifdef USE_EIGEN
void s_at2au(const Eigen::VectorXd& at, Dense& au, const Dense& FA, const int dim, const Para para);
#endif

void s_at2au(const arma::vec& at, Dense& au, const Dense& FA, const int dim, const Para para);


void s_pdapdt_lmul_fast(arma::vec& at, arma::vec& g_au, arma::vec& g_at, const arma::vec& FA, const int dim, const Para para);

void s_pdapdt_lmul_fast(arma::vec& at, Dense& g_au, arma::vec& g_at, const Dense& FA, const int dim, const Para para);

// void s_at2au_0(const Dense& at, Dense& au, const Dense& FA, const int dim = 2) {
void s_at2au_0(const double* h_at, Dense& au, const Dense& FA, const int dim, const Para para);

// void s_pdapdt_lmul(const Dense& at, const Dense& g_au, Dense& g_at, const Dense& FA, const int dim = 2) {
void s_pdapdt_lmul(const double* h_at, const Dense& g_au, double* h_gat, const Dense& FA, const int dim, const Para para);

void s_pdapdt_lmul(const Dense& at, const Dense& g_au, Dense& g_at, const Dense& FA, const int dim, const Para para);

#ifdef USE_EIGEN
void s_pdapdt_lmul(const Eigen::VectorXd& at, const Dense& g_au, Eigen::VectorXd& g_at, const Dense& FA, const int dim, const Para para);
#endif

void s_pdapdt_lmul(const arma::vec& at, const Dense& g_au, arma::vec& g_at, const Dense& FA, const int dim, const Para para);

void symmetric_tensor_assemble_indices(int*& I, int*& J, int f, int dim);

Sparse symmetric_tensor_assemble(const Dense& au, const int dim, int* I, int* J);

void symmetric_tensor_span_dot(const Dense& b, const Dense& c, Dense& p, int dim);

void symmetric_tensor_span_dot(const double* b, const double* c, double* p, int f, int dim);
