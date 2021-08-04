#include "helper.h"

arma::vec dense_array_to_arma_vec(const Dense& a) {
	arma::vec r(a.nrow());
	r.zeros();
	for (int i = 0; i < a.nrow(); i++) {
		r(i) = a(i, 0);
	}
	return r;
}

Dense arma_vec_to_dense_array(const arma::vec& a) {
	Dense r(a.n_rows, 1);
	for (int i = 0; i < a.n_rows; i++) {
		r(i, 0) = a(i);
	}
	return r;
}

Dense arma_mat_to_dense_array(const arma::mat& m) {
	Dense r(m.n_rows, m.n_cols);
	for (int j = 0; j < m.n_cols; j++) {
		for (int i = 0; i < m.n_rows; i++) {
			r(i, j) = m(i, j);
		}
	}
	return r;
}

#ifdef USE_EIGEN
Eigen::VectorXd dense_array_to_eigen_vec(const Dense& a) {
	Eigen::VectorXd r(a.nrow());
	for (int i = 0; i < a.nrow(); i++) {
		r(i) = a(i, 0);
	}
	return r;
}
#endif 