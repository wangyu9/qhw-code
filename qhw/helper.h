
#include "matrix.h"

#ifdef USE_EIGEN
#include <Eigen/Core>
#endif

#include <armadillo>

arma::vec dense_array_to_arma_vec(const Dense& a);

Dense arma_vec_to_dense_array(const arma::vec& a);

Dense arma_mat_to_dense_array(const arma::mat& m);

inline arma::mat dense_array_to_arma_mat(const Dense& D)
{
	arma::mat D_arma(D.head(), D.nrow(), D.ncol());
	return D_arma;
}

#ifdef USE_EIGEN
Eigen::VectorXd dense_array_to_eigen_vec(const Dense& a);
#endif