#pragma once

#include "cholmod_l.h"
#include <string>

cholmod_common* Begin();

int End(cholmod_common* cm);

void simple_error_handler(int status, const char* file, int line,
	const char* message);

cholmod_common* cm_default();

int configure_solve(cholmod_common* cm);

class Dense;

#define COPY_STYPE_OF_SRC_MATRIX 100

class Sparse {
public:

	Sparse(const int * I, const int * J, const double * V, const int m, const int n, const int nnz, 
			const int stype = 0, cholmod_common* cm = Begin()); 
	// stype=0 by default is unsymmetric, this will not copy lower tri to upper tri etc. 
	Sparse(cholmod_common* cm = Begin());
	~Sparse();
	Sparse(const Sparse& sparse); // copy constructor
	Sparse& operator = (const Sparse& s); // assignment operator

	static void assign(const Sparse& src, Sparse& dest, int request_type= COPY_STYPE_OF_SRC_MATRIX);
	// 100 is to copy the stype of src. 

	static void assign_value_same_pattern(const Sparse& src, Sparse& dest);
	static void sum_same_pattern(const Sparse** src, Sparse& dest, int num_src);
	static void sp_mul_same_pattern(const Sparse** src0, const Sparse** src1, const Sparse** dest, int num_src);
	
	static Sparse assemble_lap(const Sparse& GxT, const Sparse& GyT, const Sparse& GzT, const Dense& au, int dim);
	
	static Sparse assemble_lap_off_diag(const Sparse& GxT, const Sparse& GyT, const Sparse& GzT, const Sparse& GxT2, const Sparse& GyT2, const Sparse& GzT2, const Dense& au, int dim);


	void deconstruct();
	void deconstruct_keep_factor();

	void init_ori(const std::string filename, const int prefer_zomplex=0);
	void init_ori2(const std::string filename, const int prefer_zomplex = 0);

	void symbolic_factor();
	void numerical_factor();
	void solve_with_factor(const Dense& B, Dense& X) const;
	Dense solve_with_factor(const Dense& B) const;

	void solve(const Dense& B, Dense& X) const;
	Dense solve(const Dense& B) const;
	void tmul(const Dense& X, Dense& Y) const; // Y = A * X
	Dense tmul(const Dense& X) const;
	void mul(const Dense& X, Dense& Y) const; // Y = A * X
	Dense mul(const Dense& X) const;
	Dense operator * (const Dense& X) const;

	void mul2(const Dense& X, Dense& Y) const;

	// Set keep_factor to true only if mul does not change the size and sparsity pattern of Y. 
	void mul(const Sparse& B, Sparse& Y, int requested_stype=0, bool keep_factor=false) const; // 0: by default unsymmetric. // Y = A * B
	Sparse mul(const Sparse& B, int requested_stype=0) const;
	Sparse operator * (const Sparse& B) const;

	Sparse transposed() const;

	Sparse operator + (const Sparse& B) const;
	Sparse operator - (const Sparse& B) const;
	Sparse operator * (const double& m) const;

	int nrow() const;
	int ncol() const;
	int type() const;
	// int nz_at_col(int iCol) const; // removed, incorrectly implemented. 
	double norm(int type_of_norm = 1) const; /* type of norm: 0: inf. norm, 1: 1-norm, there seems to be no L2 norm */
	double my_norm(int type_of_norm = 1) const;
	double pnorm(int power) const;

	cholmod_common* cm;

	int read(const std::string filename);
	int write(const std::string filename);
	void print();

	static void check_same_pattern(const Sparse& A, const Sparse& B);

	//private:

	cholmod_sparse* sparse_data;

	static Sparse Diag(const Dense& diagm);

	static void add(const Sparse& A, const Sparse& B, Sparse& R, const double alpha = 1.0, const double beta = 1.0);
	inline static Sparse add(const Sparse& A, const Sparse& B, const double alpha = 1.0, const double beta = 1.0);

	inline static void subtract(const Sparse& A, const Sparse& B, Sparse& R);
	inline static Sparse subtract(const Sparse& A, const Sparse& B);

private:
	cholmod_factor* L; // 
};

class DenseInt {
public:
	DenseInt(int nrow = 0, int ncol = 0, const int init_value = 0);
	DenseInt(int* aa, int nrow, int ncol);
	~DenseInt();
	DenseInt(const DenseInt& dense);// copy constructor
	DenseInt& operator = (const DenseInt& t); // assignment operator

	int* head() const;

	int& operator()(int row, int col) const;

	int nrow() const;
	int ncol() const;

	int read(std::string filename, bool from_one_base_index = false);
	// todo: int write(std::string filename) const;

	friend void assign_denseint(const DenseInt& src, DenseInt& tar);

private:
	int* dense_data;
	int num_rows;
	int num_cols;
};

class Dense {
public:
	Dense(int nrow = 0, int ncol = 0, const double init_value=0, cholmod_common* cm = Begin());
	~Dense();
	Dense(const Dense& dense);// copy constructor
	Dense& operator = (const Dense& t); // assignment operator

	void deconstruct();

	int nrow() const;
	int ncol() const;
	int type() const;
	void resize(int nrow, int ncol);
	double norm(int type_of_norm=2) const; /* type of norm: 0: inf. norm, 1: 1-norm, 2: 2-norm */
	// note that X.norm(2) is the norm(X,'fro') in matlab, 
	// *NOT* the norm(X,2) which is the maximum singular value of X.  
	
	double& operator()(int row, int col) const;
	Dense operator + (const Dense& B) const;
	Dense operator - (const Dense& B) const;
	Dense operator % (const Dense& B) const;
	Dense operator * (const double& m) const;

	Dense mul(const Dense& B, bool transpose_A = false) const;
	Dense operator * (const Dense& B) const;

	Dense reduce_min(int axis=0) const; // axis=0 for now
	Dense reduce_sum(int axis) const; // axis=1 for now
	Dense reduce_norm(int axis) const; // axis=1 for now
	Dense reduce_sq_sum(int axis) const; // axis=1 for now

	double* head() const;
	// static Dense add(const Dense& A, const Dense& B); 
	double trace() const;

	void slice_assign_value(int* indices, const Dense& B, int axis = 0, int indices_length = -1);
	void slice_assign_value(const DenseInt& indices, const Dense& B, int axis = 0, int indices_length = -1);

	void initTestFor(const Sparse& A);
	int read(std::string filename);
	int write(std::string filename) const;

	void print(); 

	cholmod_common * cm;
	cholmod_dense* dense_data;

	static Dense Ones(int nrow, int ncol);
	static Dense Zeros(int nrow, int ncol);
	// static Dense Arange();

	// B <- alpha * A + B
	static void saxy(const Dense& A, Dense& B, const double alpha = 1.0);

	static void add(const Dense& A, const Dense& B, Dense& R);
	inline static Dense add(const Dense& A, const Dense& B);

	inline static void subtract(const Dense& A, const Dense& B, Dense& R);
	inline static Dense subtract(const Dense& A, const Dense& B);

	// elementwise multiplication
	static void times(const Dense& A, const Dense& B, Dense& R);
	static Dense times(const Dense& A, const Dense& B);
	
	// elementwise div
	static void div(const Dense& A, const Dense& B, Dense& R);
	static Dense div(const Dense& A, const Dense& B);

	// element-wise ops
	static void sq(const Dense& A, Dense& R);
	static Dense sq(const Dense& A);

	// not necessary for now:
	// static void times(const double c, const Dense& A, Dense& R);
	// inline static Dense times(const double c, const Dense& A);

	static void div(const double c, const Dense& A, Dense& R); // compute R <- c ./ A
	static Dense div(const double c, const Dense& A);

	// for dim(A)==dim(B) == 2 or 3
	static void cross(const Dense& A, const Dense& B, Dense& R);
	static Dense cross(const Dense& A, const Dense& B);

	static void cross2d(const Dense& A, const Dense& B, Dense& R);
	static Dense cross2d(const Dense& A, const Dense& B);

	static void mul(const Dense& A, const Dense& B, Dense& Y, bool transpose_A = false);
	static Dense mul(const Dense& A, const Dense& B, bool transpose_A = false);

	static void concatenate(const Dense& A, const Dense& B, Dense& R, const int axis=0);
	static Dense concatenate(const Dense& A, const Dense& B, const int axis = 0);


private:

};

// this return A(I(:,icol),:)



void slice_rows(const Dense& A, const DenseInt& I, Dense& R, int icol = 0);

Dense slice_rows(const Dense& A, const DenseInt& I, int icol = 0);

void slice_one_col(const Dense& A, Dense& R, int index = 0);

Dense slice_one_col(const Dense& A, int index = 0);

// B <- A * symmetric_tensor_assemble(au, dim)
void symmetric_tensor_rmul(const Sparse& A, const Dense& au, Sparse& B, int dim, int option); 

// B <- A * diag(b)
void sparse_diag_mul(const Sparse& A, const Dense& b, Sparse& B, int start_in_b);

// B[i] <- A[i] * diag(b)
void sparse_diag_mul_same_pattern(const Sparse** As, const Dense& b,
	Sparse** Bs, int* starts_in_b, int num);

// IOs

int read_int_matrix(std::string file_name, DenseInt& A, bool from_one_base_index = true);

int read_int_matrix(std::string file_name, int*& A, int& num_cols, int& num_rows, bool from_one_base_index = true);

int read_int_matrix_of_size(std::string file_name, int*& A, const int num_cols, const int num_rows, bool from_one_base_index = true);

void complementary_list(const DenseInt& known, int n, DenseInt& unknown);