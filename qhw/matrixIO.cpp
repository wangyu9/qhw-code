#include "matrix.h"

#include <stdio.h>
#include <assert.h>

#define READ_LINE_MAX 1024
FILE* read_mtx_int_header(std::string filename, int& num_rows, int& num_cols) {

    FILE* fp = fopen(filename.c_str(), "rb");

    char line[READ_LINE_MAX];
    fgets(line, READ_LINE_MAX, fp);
    // The first line is supposed to be "%%MatrixMarket matrix array real general"

    int res = fscanf(fp, "%d %d", &num_rows, &num_cols); // does not match the matrix market format. 

    assert(num_rows > 0 && num_cols > 0);

    return fp;
}

int read_mtx_int_body(int* A, FILE* fp, int& num_rows, int& num_cols, bool from_one_base_index) {

    int N = num_rows * num_cols;

    int d;

    for (int i = 0; i < N; i++) {

        if (fscanf(fp, " %d", &d) != 1)
        {
            fclose(fp);
            fprintf(
                stderr,
                "IOError: () bad format after reading %d entries\n", i);
            A = NULL;
            return -1;
        }
        A[i] = d - (from_one_base_index ? 1 : 0);
    }
    fclose(fp);
    return 0;
}

int DenseInt::read(std::string filename, bool from_one_base_index) {

    int N_old = num_rows * num_cols;

    FILE* fp = read_mtx_int_header(filename, num_rows, num_cols);

    int N = num_rows * num_cols;

    if (N != N_old) {
        delete[] dense_data;
        dense_data = new int[N];
    }

    int r = read_mtx_int_body(dense_data, fp, num_rows, num_cols, from_one_base_index);

    if (N > 3)
        printf("DenseInt::read(), matrix read (%d,%d), %d,%d,...%d\n", num_rows, num_cols,
            dense_data[0], dense_data[1], dense_data[N - 1]);

    return r;
}


int read_int_matrix_of_size(std::string file_name, int*& A, const int num_cols, const int num_rows, bool from_one_base_index) {
    int tcols, trows;
    int r = read_int_matrix(file_name, A, tcols, trows, from_one_base_index);
    if (tcols != num_cols || trows != num_rows) {
        printf("Error: read_int_matrix_of_size(): size does not match the prescribed one.\n");
    }
    return r;
}

// cannot be used for matrix market format. 
int read_int_matrix(std::string file_name, int*& A, int& num_cols, int& num_rows, bool from_one_base_index) {

    FILE* fp = fopen(file_name.c_str(), "rb");
    // int num_cols, num_rows;

    int res = fscanf(fp, "%d %d", &num_cols, &num_rows); // does not match the matrix market format. 

    assert(num_rows > 0 && num_cols > 0);

    int N = num_rows * num_cols;

    A = new int[N];
    int d;

    for (int i = 0; i < N; i++) {

        if (fscanf(fp, " %d", &d) != 1)
        {
            fclose(fp);
            fprintf(
                stderr,
                "IOError: () bad format after reading %d entries\n", i);
            A = NULL;
            return -1;
        }
        A[i] = d - (from_one_base_index ? 1 : 0);
    }

    return 0;
}

int read_int_matrix(std::string file_name, DenseInt& A, bool from_one_base_index) {
    int* aa;
    int num_cols;
    int num_rows;

    int r = read_int_matrix(file_name, aa, num_cols, num_rows, from_one_base_index);

    A = DenseInt(aa, num_rows, num_cols);

    return r;
}

void complementary_list(const DenseInt& known, int n, DenseInt& unknown) {
    bool* mask = new bool[n]; // true for known vetex indices.
    for (int i = 0; i < n; i++) {
        mask[i] = false;
    }
    for (int i = 0; i < known.nrow(); i++) {
        mask[known(i, 0)] = true;
    }

    unknown = DenseInt(n - known.nrow(), 1);

    int num_unknown = 0;
    for (int i = 0; i < n; i++) {
        if (!mask[i]) {
            unknown(num_unknown, 0) = i;
            num_unknown++;
        }
    }

    delete[] mask;
}