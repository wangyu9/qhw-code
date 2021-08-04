#include "mesh.h"
#include "matrix.h"

#include "assert.h"

// void grad_core2d(const Dense& V, const DenseInt& F, Dense& R) 
Dense grad_core2d(const Dense& V, const DenseInt& F, int p0=0, int p1=1, int p2=2)
{
	assert(V.ncol() == 2);

	const int f = F.nrow();

	Dense v0 = slice_rows(V, F, p0);
	Dense v1 = slice_rows(V, F, p1);
	Dense v2 = slice_rows(V, F, p2);

	Dense v20 = v2 - v0;
	Dense v10 = v1 - v0;

	Dense dot12 = (v20 % v10).reduce_sum(1);

	Dense ns12 = Dense::sq(Dense::cross2d(v20, v10));

	// times supports boardcast.
	Dense R = Dense::times(v10, Dense::div(Dense::times(v20, v20).reduce_sum(1), ns12))
			+ Dense::times(v20, Dense::div(Dense::times(v10, v10).reduce_sum(1), ns12))
			- Dense::times(v10+v20, Dense::div(dot12, ns12));

	return R;
}

Dense grad_core3d(const Dense& V, const DenseInt& F, int p0 = 0, int p1 = 1, int p2 = 2, int p3 = 3)
{
	assert(V.ncol() == 3);

	const int f = F.nrow();

	Dense v0 = slice_rows(V, F, p0);
	Dense v1 = slice_rows(V, F, p1);
	Dense v2 = slice_rows(V, F, p2);
	Dense v3 = slice_rows(V, F, p3);

	Dense v30 = v3 - v0;
	Dense v20 = v2 - v0;
	Dense v10 = v1 - v0;

	Dense v31 = v30 - v10;
	Dense v21 = v20 - v10;

	Dense n123 = Dense::cross(v21, v31);
	n123 = Dense::times(n123, Dense::div(1, n123.reduce_norm(1)));

	Dense g0 = Dense::times(n123, Dense::div(-1, Dense::times(n123, v10).reduce_sum(1)));

	return g0;

}

Dense area(const Dense& V, const DenseInt& F)
{
	assert(V.ncol() == 2);
	assert(F.ncol() == 3);
	
	Dense v0 = slice_rows(V, F, 0);
	Dense v1 = slice_rows(V, F, 1);
	Dense v2 = slice_rows(V, F, 2);

	Dense v20 = v2 - v0;
	Dense v10 = v1 - v0;

	return Dense::cross2d(v10, v20) * 0.5;
}

Dense volume(const Dense& V, const DenseInt& F)
{
	assert(V.ncol() == 3);
	assert(F.ncol() == 4);

	const int f = F.nrow();

	Dense v0 = slice_rows(V, F, 0);
	Dense v1 = slice_rows(V, F, 1);
	Dense v2 = slice_rows(V, F, 2);
	Dense v3 = slice_rows(V, F, 3);

	Dense v30 = v3 - v0;
	Dense v20 = v2 - v0;
	Dense v10 = v1 - v0;

	Dense s023 = Dense::cross(v20, v30);

	Dense v = Dense::times(s023, v10).reduce_sum(1) * (1.0/6.0); // it should be 1/6 rather than -1/6 here.

	return v;

}

void threshold_small_entries(Dense& a) {
	double epsilon = 1e-7;
	for (int j = 0; j < a.ncol(); j++) {
		for (int i = 0; i < a.nrow(); i++) {
			if (abs(a(i, j)) < epsilon) {
				a(i, j) = epsilon;
			}
		}
	}
}

Grad compute_grads_pre(const Dense& V, const DenseInt& F)
{
	Grad grad;

	const int dim = F.ncol() - 1;
	assert(dim == 2 || dim == 3);

	if (dim == 3) {
		grad.g0 = grad_core3d(V, F, 0, 1, 2, 3);
		grad.g1 = grad_core3d(V, F, 1, 2, 3, 0);
		grad.g2 = grad_core3d(V, F, 2, 3, 0, 1);
		grad.g3 = grad_core3d(V, F, 3, 0, 1, 2);

		threshold_small_entries(grad.g0);
		threshold_small_entries(grad.g1);
		threshold_small_entries(grad.g2);
		threshold_small_entries(grad.g3);
	} 
	else {
		grad.g0 = grad_core2d(V, F, 0, 1, 2);
		grad.g1 = grad_core2d(V, F, 1, 2, 0);
		grad.g2 = grad_core2d(V, F, 2, 0, 1);

		threshold_small_entries(grad.g0);
		threshold_small_entries(grad.g1);
		threshold_small_entries(grad.g2);
	}
	
	return grad;
}

#include "timer.h"
void compute_grads(const Dense& V, const DenseInt& F, const Grad& grad, const Dense& W, Dense& GW) {

	//const Dense& V = grad.V;
	//const DenseInt& F = grad.F;

	const int f = F.nrow();
	const int dim = F.ncol() - 1;

	if (dim == 3) {
		// f * 3, f * m
		// should not use times, but introducing a new op.

		double t = GetTime();

		// Slower: 1.02s compared to 0.9 s after being rewritten. 
		//Dense GWx =
		//	  Dense::times(slice_one_col(grad.g0, 0), slice_rows(W, F, 0))
		//	+ Dense::times(slice_one_col(grad.g1, 0), slice_rows(W, F, 1))
		//	+ Dense::times(slice_one_col(grad.g2, 0), slice_rows(W, F, 2))
		//	+ Dense::times(slice_one_col(grad.g3, 0), slice_rows(W, F, 3));

		Dense GWx = Dense::times(slice_one_col(grad.g0, 0), slice_rows(W, F, 0));
		Dense::saxy(Dense::times(slice_one_col(grad.g1, 0), slice_rows(W, F, 1)), GWx);
		Dense::saxy(Dense::times(slice_one_col(grad.g2, 0), slice_rows(W, F, 2)), GWx);
		Dense::saxy(Dense::times(slice_one_col(grad.g3, 0), slice_rows(W, F, 3)), GWx);

		printf("compute_grads(): check point 1: %f\n", GetTime() - t);

		//Dense GWy =
		//	  Dense::times(slice_one_col(grad.g0, 1), slice_rows(W, F, 0))
		//	+ Dense::times(slice_one_col(grad.g1, 1), slice_rows(W, F, 1))
		//	+ Dense::times(slice_one_col(grad.g2, 1), slice_rows(W, F, 2))
		//	+ Dense::times(slice_one_col(grad.g3, 1), slice_rows(W, F, 3));

		Dense GWy = Dense::times(slice_one_col(grad.g0, 1), slice_rows(W, F, 0));
		Dense::saxy(Dense::times(slice_one_col(grad.g1, 1), slice_rows(W, F, 1)), GWy);
		Dense::saxy(Dense::times(slice_one_col(grad.g2, 1), slice_rows(W, F, 2)), GWy);
		Dense::saxy(Dense::times(slice_one_col(grad.g3, 1), slice_rows(W, F, 3)), GWy);

		//Dense GWz =
		//	  Dense::times(slice_one_col(grad.g0, 2), slice_rows(W, F, 0))
		//	+ Dense::times(slice_one_col(grad.g1, 2), slice_rows(W, F, 1))
		//	+ Dense::times(slice_one_col(grad.g2, 2), slice_rows(W, F, 2))
		//	+ Dense::times(slice_one_col(grad.g3, 2), slice_rows(W, F, 3));

		Dense GWz = Dense::times(slice_one_col(grad.g0, 2), slice_rows(W, F, 0));
		Dense::saxy(Dense::times(slice_one_col(grad.g1, 2), slice_rows(W, F, 1)), GWz);
		Dense::saxy(Dense::times(slice_one_col(grad.g2, 2), slice_rows(W, F, 2)), GWz);
		Dense::saxy(Dense::times(slice_one_col(grad.g3, 2), slice_rows(W, F, 3)), GWz);

		printf("compute_grads(): check point 5: %f\n", GetTime() - t);

		GW = Dense::concatenate(Dense::concatenate(GWx, GWy), GWz);

		printf("compute_grads(): check point 6: %f\n", GetTime() - t);
	}
	else {

		//Dense GWx =
		//	  Dense::times(slice_one_col(grad.g0, 0), slice_rows(W, F, 0))
		//	+ Dense::times(slice_one_col(grad.g1, 0), slice_rows(W, F, 1))
		//	+ Dense::times(slice_one_col(grad.g2, 0), slice_rows(W, F, 2));

		Dense GWx = Dense::times(slice_one_col(grad.g0, 0), slice_rows(W, F, 0));
		Dense::saxy(Dense::times(slice_one_col(grad.g1, 0), slice_rows(W, F, 1)), GWx);
		Dense::saxy(Dense::times(slice_one_col(grad.g2, 0), slice_rows(W, F, 2)), GWx);

		//Dense GWy =
		//	  Dense::times(slice_one_col(grad.g0, 1), slice_rows(W, F, 0))
		//	+ Dense::times(slice_one_col(grad.g1, 1), slice_rows(W, F, 1))
		//	+ Dense::times(slice_one_col(grad.g2, 1), slice_rows(W, F, 2));

		Dense GWy = Dense::times(slice_one_col(grad.g0, 1), slice_rows(W, F, 0));
		Dense::saxy(Dense::times(slice_one_col(grad.g1, 1), slice_rows(W, F, 1)), GWy);
		Dense::saxy(Dense::times(slice_one_col(grad.g2, 1), slice_rows(W, F, 2)), GWy);

		GW = Dense::concatenate(GWx, GWy);
	}

}


void copy_matrix_to_tf_array_float(const Dense& V, TF_Tensor* atf) {
	for (int i = 0; i < V.nrow(); i++) {
		for (int j = 0; j < V.ncol(); j++) {
			// tf data is row-major.
			int index = i * V.ncol() + j;
			static_cast<float*>(TF_TensorData(atf))[index] = V(i, j);
		}
	}
}

void copy_matrix_to_tf_array_int(const DenseInt& F, TF_Tensor* atf) {
	for (int i = 0; i < F.nrow(); i++) {
		for (int j = 0; j < F.ncol(); j++) {
			// tf data is row-major.
			int index = i * F.ncol() + j;
			static_cast<int*>(TF_TensorData(atf))[index] = F(i, j);
		}
	}
}

void copy_tf_array_to_matrix_float(TF_Tensor* atf, Dense& V) {
	float* AA = static_cast<float*>(TF_TensorData(atf));
	for (int i = 0; i < V.nrow(); i++) {
		for (int j = 0; j < V.ncol(); j++) {
			// tf data is row-major.
			int index = i * V.ncol() + j;
			V(i, j) = (double) AA[index];
		}
	}
}

void copy_matrix_to_tf_array_double(const Dense& V, TF_Tensor* atf) {
	for (int i = 0; i < V.nrow(); i++) {
		for (int j = 0; j < V.ncol(); j++) {
			// tf data is row-major.
			int index = i * V.ncol() + j;
			static_cast<double*>(TF_TensorData(atf))[index] = V(i, j);
		}
	}
}

void copy_matrix_to_tf_array_int64(const DenseInt& F, TF_Tensor* atf) {
	for (int i = 0; i < F.nrow(); i++) {
		for (int j = 0; j < F.ncol(); j++) {
			// tf data is row-major.
			int index = i * F.ncol() + j;
			static_cast<int64_t*>(TF_TensorData(atf))[index] = F(i, j);
		}
	}
}

void copy_tf_array_to_matrix_double(TF_Tensor* atf, Dense& V) {
	double* AA = static_cast<double*>(TF_TensorData(atf));
	for (int i = 0; i < V.nrow(); i++) {
		for (int j = 0; j < V.ncol(); j++) {
			// tf data is row-major.
			int index = i * V.ncol() + j;
			V(i, j) = AA[index];
		}
	}
}

void GRADtf::run(const Dense& W, Dense& GW) 
{
	int64_t dims_W[2] = { W.nrow(), W.ncol() };

	input_tensors[2] = TF_AllocateTensor(TF_FLOAT, dims_W, 2, dims_W[0] * dims_W[1] * sizeof(float));

	copy_matrix_to_tf_array_float(W, input_tensors[2]);

	TF_Tensor* out = (*model)(input_tensors, 3);

	GW = Dense::Ones(f * dim, W.ncol());

	copy_tf_array_to_matrix_float(out, GW);
}

GRADtf::GRADtf(const Dense& V, const DenseInt& F, int ncolW) 
{
	f = F.nrow();
	dim = F.ncol() - 1;

	auto path = std::string("./");
	model = (dim == 3) ?
		new Model(ReadBinaryProto(path + "tf_graphs/compute_grads_float.pb"), { "V", "F", "W" }, "GW") :
		new Model(ReadBinaryProto(path + "tf_graphs/compute_grads_2d_float.pb"), { "V", "F", "W" }, "GW");

	int64_t dims_V[2] = { V.nrow(), V.ncol() };
	int64_t dims_F[2] = { F.nrow(), F.ncol() };
	int64_t dims_W[2] = { V.nrow(), ncolW };


	input_tensors = new TF_Tensor*[3];
	input_tensors[0] = TF_AllocateTensor(TF_FLOAT, dims_V, 2, V.nrow() * V.ncol() * sizeof(float));
	input_tensors[1] = TF_AllocateTensor(TF_INT32, dims_F, 2, F.nrow() * F.ncol() * sizeof(int));
	input_tensors[2] = NULL;

	copy_matrix_to_tf_array_float(V, input_tensors[0]);
	copy_matrix_to_tf_array_int(F, input_tensors[1]);
}


void tf_compute_grads(const Dense& V, const DenseInt& F, const Dense& W, Dense& GW)
{

	const int f = F.nrow();
	const int dim = F.ncol() - 1;

	assert(dim == 3 || dim == 2);

	auto path = std::string("/media/ssd1/wangyu/WorkSpace/linsolve/apps/");
	Model M = (dim == 3) ?
		Model(ReadBinaryProto(path + "tf_graphs/compute_grads_float.pb"), { "V", "F", "W" }, "GW") :
		Model(ReadBinaryProto(path + "tf_graphs/compute_grads_2d_float.pb"), { "V", "F", "W" }, "GW");

	int64_t dims_V[2] = { V.nrow(), V.ncol() };
	int64_t dims_F[2] = { F.nrow(), F.ncol() };
	int64_t dims_W[2] = { W.nrow(), W.ncol() };

	TF_Tensor* input_tensors[3] = {
		TF_AllocateTensor(TF_FLOAT, dims_V, 2, V.nrow() * V.ncol() * sizeof(float)),
		TF_AllocateTensor(TF_INT32, dims_F, 2, F.nrow() * F.ncol() * sizeof(int)),
		TF_AllocateTensor(TF_FLOAT, dims_W, 2, W.nrow() * W.ncol() * sizeof(float))
	};

	copy_matrix_to_tf_array_float(V, input_tensors[0]);
	copy_matrix_to_tf_array_int(F, input_tensors[1]);
	copy_matrix_to_tf_array_float(W, input_tensors[2]);

	TF_Tensor* out = M(input_tensors, 3);

	GW = Dense::Ones(f * dim, W.ncol());

	copy_tf_array_to_matrix_float(out, GW);
}


void GRADtf64::run(const Dense& W, Dense& GW)
{
	int64_t dims_W[2] = { W.nrow(), W.ncol() };

	input_tensors[2] = TF_AllocateTensor(TF_DOUBLE, dims_W, 2, dims_W[0] * dims_W[1] * sizeof(double));

	copy_matrix_to_tf_array_double(W, input_tensors[2]);

	TF_Tensor* out = (*model)(input_tensors, 3);

	GW = Dense::Ones(f * dim, W.ncol());

	copy_tf_array_to_matrix_double(out, GW);
}

GRADtf64::GRADtf64(const Dense& V, const DenseInt& F, int ncolW)
{
	f = F.nrow();
	dim = F.ncol() - 1;

	auto path = std::string("./");
	model = (dim==3) ? 
		new Model(ReadBinaryProto(path + "tf_graphs/compute_grads_double.pb"), { "V", "F", "W" }, "GW") :
		new Model(ReadBinaryProto(path + "tf_graphs/compute_grads_2d_double.pb"), { "V", "F", "W" }, "GW");

	int64_t dims_V[2] = { V.nrow(), V.ncol() };
	int64_t dims_F[2] = { F.nrow(), F.ncol() };
	int64_t dims_W[2] = { V.nrow(), ncolW };


	input_tensors = new TF_Tensor * [3];
	input_tensors[0] = TF_AllocateTensor(TF_DOUBLE, dims_V, 2, V.nrow() * V.ncol() * sizeof(double));
	input_tensors[1] = TF_AllocateTensor(TF_INT64, dims_F, 2, F.nrow() * F.ncol() * sizeof(int64_t));
	input_tensors[2] = NULL;

	copy_matrix_to_tf_array_double(V, input_tensors[0]);
	copy_matrix_to_tf_array_int64(F, input_tensors[1]);
}

#include <vector>
#include <algorithm>
#include <iterator>
#include <functional>

void extract(int * FL, double * VL, bool* mask, int *porder, int f, int n) {
	std::vector<int> uu, kk;
	for (int i = 0; i < f; i++) {
		mask[FL[i]];
	}
}

void sparse_grads(const Dense& V, const DenseInt& F,
	const DenseInt& known, // known may not be sorted.
	Sparse& Gx, Sparse& Gy,
	Sparse& Gz, 
	Sparse& Gxk, Sparse& Gxu, Sparse& Gyk, Sparse& Gyu,
	Sparse& Gzk, Sparse& Gzu)
{
	const int dim = F.ncol() - 1;
	assert(dim == 2 || dim == 3);

	const int n = V.nrow();
	const int f = F.nrow();

	bool* mask = new bool[n]; // true for known vetex indices.
	for (int i = 0; i < n; i++) {
		mask[i] = false;
	}
	for (int i = 0; i < known.nrow(); i++) {
		mask[known(i, 0)] = true;
	}

	int* porder = new int[n];
	for (int i = 0; i < n; i++) {
		porder[i] = -1;
	}

	int total_known = 0;
	int total_unknown = 0;

	// since known is not sorted, but unknonw is always sorted as how we generated. 
	for (int i = 0; i < known.nrow(); i++) {
		porder[known(i, 0)] = total_known;
		total_known++;
	}

	for (int i = 0; i < n; i++) {
		if (mask[i]) {
			;
		}
		else {
			porder[i] = total_unknown;
			total_unknown++;
		}
	}

	printf("#known=%d, #unknown=%d\n", total_known, total_unknown);

	Dense g0, g1, g2, g3;

	{
		if (dim == 3) {
			g0 = grad_core3d(V, F, 0, 1, 2, 3);
			g1 = grad_core3d(V, F, 1, 2, 3, 0);
			g2 = grad_core3d(V, F, 2, 3, 0, 1);
			g3 = grad_core3d(V, F, 3, 0, 1, 2);
		}
		else {
			g0 = grad_core2d(V, F, 0, 1, 2);
			g1 = grad_core2d(V, F, 1, 2, 0);
			g2 = grad_core2d(V, F, 2, 0, 1);
		}

		// This might be important, but seems fine without it:   
		// making sure that g's does not contain zeros values. 
		// so the sparse matrix will have exactly the same sparsity pattern. 
		threshold_small_entries(g0);
		threshold_small_entries(g1);
		threshold_small_entries(g2);
		if (dim == 3) {
		threshold_small_entries(g3);
		}

		// printf("sparse_grads(): norms:%f,%f,%f,%f\n", g0.norm(), g1.norm(), g2.norm(), g3.norm());

		int* linf = new int[f];
		for (int i = 0; i < f; i++) {
			linf[i] = i;
		}

		if (dim == 3) {

			Gx = 
				  Sparse(linf, &F(0, 0), &g0(0, 0), f, n, f)
				+ Sparse(linf, &F(0, 1), &g1(0, 0), f, n, f)
				+ Sparse(linf, &F(0, 2), &g2(0, 0), f, n, f)
				+ Sparse(linf, &F(0, 3), &g3(0, 0), f, n, f);

			Gy =
				  Sparse(linf, &F(0, 0), &g0(0, 1), f, n, f)
				+ Sparse(linf, &F(0, 1), &g1(0, 1), f, n, f)
				+ Sparse(linf, &F(0, 2), &g2(0, 1), f, n, f)
				+ Sparse(linf, &F(0, 3), &g3(0, 1), f, n, f);

			Gz =
				  Sparse(linf, &F(0, 0), &g0(0, 2), f, n, f)
				+ Sparse(linf, &F(0, 1), &g1(0, 2), f, n, f)
				+ Sparse(linf, &F(0, 2), &g2(0, 2), f, n, f)
				+ Sparse(linf, &F(0, 3), &g3(0, 2), f, n, f);
		} 
		else {

			Gx =
				  Sparse(linf, &F(0, 0), &g0(0, 0), f, n, f)
				+ Sparse(linf, &F(0, 1), &g1(0, 0), f, n, f)
				+ Sparse(linf, &F(0, 2), &g2(0, 0), f, n, f);

			Gy =
				  Sparse(linf, &F(0, 0), &g0(0, 1), f, n, f)
				+ Sparse(linf, &F(0, 1), &g1(0, 1), f, n, f)
				+ Sparse(linf, &F(0, 2), &g2(0, 1), f, n, f);

		}

		// onnz: original nnz, i.e. length of OI, OJ, OV.
		// mask, porder, are also used
		std::function<Sparse(int*, int*, double*, int, int, int,  bool)> SparseColSlice 
			= [&](int* OI, int* OJ, double* OV, int onnz, int nm, int nn, bool flip_mask)
		{
			int nnnz = 0; // nnz in the new, sliced matrix.
			for (int i = 0; i < onnz; i++) {
				// mask[OJ[i]]: 1: keep in the list, 0: not keep.
				if (mask[OJ[i]] ^ flip_mask) // xor
					nnnz++;
			}

			int* NI = new int[nnnz];
			int* NJ = new int[nnnz];
			double* NV = new double[nnnz];

			int cnnz = 0;
			for (int i = 0; i < onnz; i++) {
				if (mask[OJ[i]] ^ flip_mask) {
					NI[cnnz] = OI[i];
					NJ[cnnz] = porder[OJ[i]]; // the new index of J.
					NV[cnnz] = OV[i];
					cnnz++;
				}
			}

			Sparse M(NI, NJ, NV, nm, nn, nnnz);

			delete[] NI;
			delete[] NJ;
			delete[] NV;

			return M;
		};

		if (dim == 3) {

			Gxk =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 0), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 0), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 0), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 3), &g3(0, 0), f, f, total_known, false);

			Gyk =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 1), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 1), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 1), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 3), &g3(0, 1), f, f, total_known, false);

			Gzk =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 2), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 2), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 2), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 3), &g3(0, 2), f, f, total_known, false);

			Gxu =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 0), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 0), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 0), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 3), &g3(0, 0), f, f, total_unknown, true);

			Gyu =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 1), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 1), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 1), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 3), &g3(0, 1), f, f, total_unknown, true);

			Gzu =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 2), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 2), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 2), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 3), &g3(0, 2), f, f, total_unknown, true);
		}
		else {

			Gxk =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 0), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 0), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 0), f, f, total_known, false);

			Gyk =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 1), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 1), f, f, total_known, false)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 1), f, f, total_known, false);

			Gxu =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 0), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 0), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 0), f, f, total_unknown, true);

			Gyu =
				  SparseColSlice(linf, &F(0, 0), &g0(0, 1), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 1), &g1(0, 1), f, f, total_unknown, true)
				+ SparseColSlice(linf, &F(0, 2), &g2(0, 1), f, f, total_unknown, true);

		}

		delete[] linf;

		printf("sparse_grads(): Size, Gxu:(%d,%d), Gzk:(%d,%d)\n", Gxu.nrow(), Gxu.ncol(),Gzk.nrow(), Gzk.ncol());

	}

	delete[] mask;
	delete[] porder;

}