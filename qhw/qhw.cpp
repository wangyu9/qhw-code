#ifdef USE_EIGEN
#define EIGEN_USE_MKL_ALL
#endif
// do this before include Eigen.

#include "optim.hpp"

#include "alap.h"
#include "matrix.h" // ""

#include "timer.h"

#ifdef USE_EIGEN
#include <Eigen/Core>
#endif

#include <iostream>

#ifdef USE_LBFGSPP
#include <LBFGS.h>
#endif

#include <iostream>

#include "helper.h"
#include "mesh.h"

static void show_usage()
{
    std::cerr << "Usage: " 
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t-d,--destination DESTINATION\tSpecify the destination path"
              << std::endl;
}

/*
Differentiable Projection of Weights
*/

arma::mat W2F_n(const arma::mat& W)
{
    using namespace arma;

    arma::mat SW = sum(W, 1); // dim=1 so it sums all cols for each row. 

    arma::mat FW = W.each_col() / SW;

    return FW;
}

const double DEFAULT_W_EPSILON = 1e-2;

arma::mat W2F_m1(const arma::mat& W, const double epsilon=DEFAULT_W_EPSILON)
{
    // In matlab grammar: W2F_m = @(W) (W>epsilon) .* W + (W<=epsilon) .* (- W.^3 / epsilon^2  + 2 * W.^2 / epsilon );

    using namespace arma;

    arma::mat FW = (W > epsilon) % W + (W <= epsilon) % (- W % W % W / (epsilon*epsilon) + 2 * W % W / epsilon);

    return FW;
}

arma::mat dFW2dW_m1(const arma::mat& D, const arma::mat& W, const double epsilon=DEFAULT_W_EPSILON)
{
    // In matlab grammar: dFW2dW_m = @(D,W) ( (W>epsilon) + (W<=epsilon) .* (- 3 * W.^2 / epsilon^2  + 4 * W / epsilon ) ) .* D;

    using namespace arma;

    arma::mat FD = ( (W > epsilon) + (W <= epsilon) % (-3 * W % W / (epsilon*epsilon) + 4 * W / epsilon) ) % D; 

    return FD;
}

const double DEFAULT_W_EPSILON1 = 0e-2;
const double DEFAULT_W_EPSILON2 = 1e-4;

arma::mat W2F_m2(const arma::mat& W, const double e1 = DEFAULT_W_EPSILON1, const double e2 = DEFAULT_W_EPSILON2)
{

    using namespace arma;

    double c1 = (e1 + e2) / ((e2 - e1) * (e2 - e1) * (e2 - e1));
    double c2 = (e1 + 2 * e2) / ((e2 - e1) * (e2 - e1));

    arma::mat FW = (W <= -1000) % W -(W > e1) % (W < e2) % pow(W - e1, 3) * c1 + (W > e1) % (W < e2) % pow(W - e1, 2) * c2 + (W >= e2) % W;

    return FW;
}

arma::mat dFW2dW_m2(const arma::mat& D, const arma::mat& W, const double e1 = DEFAULT_W_EPSILON1, const double e2 = DEFAULT_W_EPSILON2)
{

    using namespace arma;

    double c1 = (e1 + e2) / ((e2 - e1) * (e2 - e1) * (e2 - e1));
    double c2 = (e1 + 2 * e2) / ((e2 - e1) * (e2 - e1));

    arma::mat FD = ((W <= -1000) - (W > e1) % (W < e2) % pow(W - e1, 2) * 3 * c1 + (W > e1) % (W < e2) % pow(W - e1, 2) * c2 + (W >= e2) ) % D;

    return FD;
}

arma::mat dFW2dW_n(const arma::mat& D, const arma::mat& W)
{
    // In matlab grammar: dFW2dW_n = @(D,W) bsxfun(@rdivide, D, sum(W,2)) - bsxfun(@times, bsxfun(@rdivide, W, sum(W,2).^2), sum(D,2));

    using namespace arma;

    arma::mat FD;

    arma::mat SW = sum(W, 1);
    arma::mat SD = sum(D, 1);

    arma::mat MM = W.each_col() / (SW % SW);

    FD = D.each_col() / SW - MM.each_col() % SD; // % is the element-wise multiplication, NOT '*'!!!	 

    return FD;
}

Dense W2F_n(const Dense& W)
{
    // const arma::mat W_arma(W.head(), W.nrow(), W.ncol());
    const arma::mat W_arma = dense_array_to_arma_mat(W);

    arma::mat FW_arma = W2F_n(W_arma);

    Dense FW = arma_mat_to_dense_array(FW_arma);

    return FW;
}

Dense dFW2dW_n(const Dense& D, const Dense& W)
{

    const arma::mat W_arma = dense_array_to_arma_mat(W);
    const arma::mat D_arma = dense_array_to_arma_mat(D);

    arma::mat FD = dFW2dW_n(D_arma, W_arma);

    return arma_mat_to_dense_array(FD);
}

Dense W2F_m(const Dense& W)
{
    const arma::mat W_arma = dense_array_to_arma_mat(W);

    arma::mat FW_arma = W2F_m2(W_arma);

    Dense FW = arma_mat_to_dense_array(FW_arma);

    return FW;
}

Dense dFW2dW_m(const Dense& D, const Dense& W)
{

    const arma::mat W_arma = dense_array_to_arma_mat(W);
    const arma::mat D_arma = dense_array_to_arma_mat(D);

    arma::mat FD = dFW2dW_m2(D_arma, W_arma);

    return arma_mat_to_dense_array(FD);
}


int main(int argc, char** argv)
{

    const std::string SNAPSHOT_NONE = std::string("none");

    std::vector <std::string> sources;
    std::string EXAMPLE = std::string("/qhw/qhw/data/tibiman-H");
    std::string SOLVER = std::string("adamd");
    std::string SNAPSHOT = SNAPSHOT_NONE;
    std::string OUTPUT = std::string("");
    int NUM_ITER = 50;
    double STEP_SIZE = 0.1;
    bool PROJECT_SIMPLEX = false;
    bool TIMING = false;
    int REPEAT = 5;
    bool LOG_HISTORY = false;
    int LBFGS_M = 10;
    double COND_BOUND = 0.2;
    double DELTA = 0;
    int verbose = 3;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage();
            return 0;
        }
        else if ((arg == "--project")) {
            PROJECT_SIMPLEX = true;
        }
        else if ((arg == "--timing")) {
            TIMING = true;
        }
        else if ((arg == "-e") || (arg == "--example")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                EXAMPLE = std::string(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--example option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--snapshot")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                SNAPSHOT = std::string(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--snapshot option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--output")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                OUTPUT = std::string(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--output option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--solver")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                SOLVER = std::string(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--solver option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--lbfgs_m")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                LBFGS_M = atoi(argv[++i]);
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--lbfgs_m option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "-n") || (arg == "--num_iter")) {
            if (i + 1 < argc) { 
                NUM_ITER = atoi(argv[++i]);
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--num_iter option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--repeat")) {
            if (i + 1 < argc) {
                REPEAT = atoi(argv[++i]);
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--repeat option requires one argument." << std::endl;
                return 1;
            }
        }
        else if (arg == "--verbose") {
            if (i + 1 < argc) {
                verbose = atoi(argv[++i]);
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--verbose option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--log_history")) {
            LOG_HISTORY = true;
        }
        else if ( (arg == "--step_size")) {
            if (i + 1 < argc) {
                STEP_SIZE = atof(argv[++i]);
                printf("step_size=%f\n", STEP_SIZE);
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--step_size option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--cond_bound")) {
            if (i + 1 < argc) {
                COND_BOUND = atof(argv[++i]);
                printf("cond_bound=%f\n", COND_BOUND);
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--cond_bound option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "--delta")) {
            if (i + 1 < argc) {
                DELTA = atof(argv[++i]);
                printf("delta=%f\n", DELTA);
            }
            else { // Uh-oh, there was no argument to the destination option.
                std::cerr << "--delta option requires one argument." << std::endl;
                return 1;
            }
        }
        else {
            sources.push_back(argv[i]);
        }
    }

    double t = 0;

    std::string folder;
    int dim = 2;

    folder = EXAMPLE + std::string("/");

    printf("Loading files from %s.\n", folder.c_str());

    if (OUTPUT.length() > 0) {
        OUTPUT = OUTPUT;
    }
    
    int length_his = NUM_ITER * (REPEAT + 1) * 2 + 1; // This *2 should not be need, just for redundancy.
    double* energy_his = new double[length_his];
    double* time_his = new double[length_his];

    cholmod_common* cm = Begin();

    Dense V;
    DenseInt F;

    Dense FA;

//#ifdef USE_LARGER_G
    Sparse Gk, Gu, G;
//#endif

    Sparse Gx, Gy, Gz;

    Sparse Gxk, Gxu, Gyk, Gyu;
    Sparse Gzk, Gzu; // for 3D

    DenseInt known;
    DenseInt unknown;
    int nk, nu; 

    Sparse L; // Note L is not p.d., just p.s.d.
    Sparse Mass;
    Sparse invMass;

    Dense mass_vertex;
    mass_vertex.read(folder + "mv.mtx");

    int f, mcdim;

    Dense BC(0, 1);
    BC.read(folder + "BC.mtx");
    printf("BC size (%d,%d)\n", BC.nrow(), BC.ncol());

    int nr;
    known.read(folder + "B.mtx", true); // setting 'true' will delete 1 from the matrix for 0-index. 

    nk = known.nrow();
    assert(known.ncol() == 1);

    {

        V.read(folder + "V.mtx");
        F.read(folder + "F.mtx", true); // from 1-index matrix.

        printf("Files loaded successfully!\n");

        f = F.nrow();
        dim = F.ncol() - 1;
        
        complementary_list(known, V.nrow(), unknown);

        FA = (dim == 3) ? volume(V, F) : area(V, F);

        sparse_grads(V, F,
            known,
            Gx, Gy, Gz,
            Gxk, Gxu, Gyk, Gyu,
            Gzk, Gzu);

        Mass = Sparse::Diag(mass_vertex);
        invMass = Sparse::Diag(Dense::div(1.0, mass_vertex));

    }

    mcdim = (dim == 2) ? 3 : 6; // so au is of dim (f*mcdim)


    Sparse GxkT = Gxk.transposed();
    Sparse GxuT = Gxu.transposed();
    Sparse GykT = Gyk.transposed();
    Sparse GyuT = Gyu.transposed();
    Sparse GzkT;
    Sparse GzuT;
    if (dim == 3) {
        GzkT = Gzk.transposed();
        GzuT = Gzu.transposed();
    }

    Sparse GxuT_A00 = GxuT;
    Sparse GxuT_A01 = GxuT;

    Sparse GyuT_A01 = GyuT;
    Sparse GyuT_A11 = GyuT;

    // Used for 3D only: 
    Sparse GxuT_A02 = GxuT;
    Sparse GzuT_A02 = GzuT;
    Sparse GyuT_A12 = GyuT;
    Sparse GzuT_A12 = GzuT;
    Sparse GzuT_A22 = GzuT;

    Sparse Mf = Sparse::Diag(FA);

    Dense Zf = Dense::Zeros(f, 1);

    Dense RFA = Dense::concatenate(FA,
        Dense::concatenate(Zf, FA));

    int stype = 0; // 0; // 0; unsymmetric // 1: symmetric and use triu

    stype = 1; // 1: symmetric and use triu
    Sparse A = GxuT_A00.mul(Gxu, stype)
        + GyuT_A01.mul(Gxu, stype)
        + GxuT_A01.mul(Gyu, stype)
        + GyuT_A11.mul(Gyu, stype);
    if (dim == 3) {
        A = A
            + GzuT_A22.mul(Gzu, stype)
            + GxuT_A02.mul(Gzu, stype)
            + GzuT_A02.mul(Gxu, stype)
            + GyuT_A12.mul(Gzu, stype)
            + GzuT_A12.mul(Gyu, stype);
    }


    L = (Mf * Gx).transposed().mul(Gx, stype)
        + (Mf * Gy).transposed().mul(Gy, stype);
    if (dim == 3) {
        L = L + (Mf * Gz).transposed().mul(Gz, stype);
    }

    int* I;
    int* J;
    symmetric_tensor_assemble_indices(I, J, f, dim);

    // Setup the objective function:
    Sparse Q; 
    stype = 1; // 1: symmetric and use triu
    (L * invMass).mul(L, Q, stype);
    // So Q is a symmetric matrix such that 
    // Q==L*M^-1*L. 

    Sparse diagm = Sparse::Diag(
        Dense::concatenate(FA, dim==2? FA : Dense::concatenate(FA, FA))
    );

    Sparse Quu, Quk;
    Sparse Qku, Qkk;

    Sparse Lua = (GxuT * Mf * Gx + GyuT * Mf * Gy);
    Sparse Lka = (GxkT * Mf * Gx + GykT * Mf * Gy);
    if (dim == 3) {
        Lua = Lua + GzuT * Mf * Gz;
        Lka = Lka + GzkT * Mf * Gz;
    }
    Sparse Lau = Lua.transposed();
    Sparse Lak = Lka.transposed();
    (Lua* invMass).mul(Lau, Quu, 0);
    (Lka* invMass).mul(Lak, Qkk, 0);
    (Lua* invMass).mul(Lak, Quk, 0);

    Qku = Quk.transposed();

    Sparse SA = GxuT * Mf * Gxk + GyuT * Mf * Gyk;
    if (dim == 3)
        SA = SA + GzuT * Mf * Gzk;
    Dense B = (SA * BC) * (-1);
    

    Dense X(B.nrow(), B.ncol());
    
    if (verbose>2) 
    {
        t = GetTime();
        for (int i = 0; i < 1; i++) 
        {
            X = A.mul(B); 
        }
        printf("Time for linear matrix-vector mul :%f.\n", GetTime() - t);

        Dense X3(B.nrow(), B.ncol());
        t = GetTime();
        for (int i = 0; i < 1000; i++) 
        {
            Dense::saxy(X, X3, -0.4);
        }
        printf("Time for saxy x1000:%f.\n", GetTime() - t);

        t = GetTime();
        configure_solve(A.cm);
        A.solve(B, X); // X = A \ B;
        printf("Time for one linear solve :%f.\n", GetTime() - t);
    }

    Dense Y = A.mul(X);

    t = GetTime();
    A.symbolic_factor();
    printf("symbolic_factor: %f seconds.\n", GetTime() - t);

    t = GetTime();
    A.numerical_factor();
    printf("numerical_factor: %f seconds.\n", GetTime() - t);


    Dense W_u;
    Dense GW;
    Dense Res_u;
    Dense BR;
    Dense PS;

    Dense st0 = Dense::Ones(f, 1);
    // Dense st0 = FA;

    Dense at0 = Dense::concatenate(st0,
        Dense::concatenate(st0 * 1e-6, st0));

    if (dim == 3) 
    {
        at0 = Dense::concatenate(at0,
            Dense::concatenate(Dense::concatenate(st0,st0) * 1e-6, st0));
    }

    if (SNAPSHOT != SNAPSHOT_NONE) 
    {
        Dense at0_snapshot; 
        at0_snapshot.read(SNAPSHOT);
        if (at0.nrow() == at0_snapshot.nrow() && at0.ncol() == at0_snapshot.ncol() ) {
            printf("Snapshot file %s is loaded.\n", SNAPSHOT.c_str());
            at0 = at0_snapshot;
        }
        else {
            printf("Snapshot file is of wrong size! File ignored. \n");
        }
    }
 

    Dense au = at0;

    Sparse diagAU = diagm;

    Dense W(L.nrow(), BC.ncol());
    W.slice_assign_value(known, BC, 0); // W(known, :) = BC;

    Dense FW(L.nrow(), BC.ncol());
    FW.slice_assign_value(known, BC, 0);

    Dense FW_u, FW_k;

    Dense gau = Dense::Zeros(f * mcdim, 1);
    Dense gau_j = Dense::Zeros(f * mcdim, 1);

    t = GetTime();

    std::function<Dense(const Dense&)> W2F_pn = [&](const Dense& W)
    {
        return W2F_n(W2F_m(W));
        //return W2F_n(W);
    };

    std::function<Dense(const Dense&, const Dense&)> dFW2dW_pn = [&](const Dense& D, const Dense& W)
    {
        return dFW2dW_n(dFW2dW_m(D, W), W2F_m(W));
        //return dFW2dW_n(D, W);
    };

    std::function<Dense(const Dense&)> W2F;
    std::function<Dense(const Dense&, const Dense&)> dFW2dW;
    
    if (PROJECT_SIMPLEX) 
    {
        W2F = W2F_pn;
        dFW2dW = dFW2dW_pn;
    }
    else 
    {
        W2F = [&](const Dense& W)
        {
            return W;
        };
        dFW2dW = [&](const Dense& D, const Dense& W)
        {
            return D;
        };
    }

    Grad grad = compute_grads_pre(V, F);

    // GRADtf64 grad_tf = GRADtf64(V, F, W.ncol());
    GRADtf grad_tf = GRADtf(V, F, W.ncol());

    int i = 0;
    double start_time = GetTime();
    // Dense gat = Dense::Zeros(mcdim * f, 1);
    // Dense at = Dense::concatenate(Dense::Ones(f, 1),
    //     Dense::concatenate(Dense::Ones(f, 1) * 1e-6, Dense::Ones(f, 1))); // do not use Dense::Zeros(f, 1)

    const Para para; 

    auto fun = [&](const arma::vec& at, arma::vec& gat)
    {

        t = GetTime();

        if (TIMING) 
        {
            printf("Entered loop for timing.\n");
        }

        /* update au from at */

        // arma::vec at_tmp = at;
        // s_at2au_fast(at_tmp, au, FA, dim, para);
        s_at2au(at, au, FA, dim, para); 

        if (TIMING) 
        {
            printf("Check point 1: %f.\t Apply Parameterization.\n", GetTime() - t); t = GetTime();
        }

        int stype = 1; // 1: symmetric and use triu

        Sparse A2 = Sparse::assemble_lap(GxuT, GyuT, GzuT, au, dim);
        Sparse::assign_value_same_pattern(A2, A);
        // A.symbolic_factor(); // no need since sparsity pattern remain unchanged. 

        B = Sparse::assemble_lap_off_diag(GxuT, GyuT, GzuT, GxkT, GykT, GzkT, au, dim) * (BC * (-1));

        if (TIMING) 
        {
            printf("Check point 2: %f.\t Assemble Lap.\n", GetTime() - t); t = GetTime();
        }

        configure_solve(A.cm);

        A.numerical_factor();

        if (TIMING) 
        {
            printf("Check point 3: %f.\t Numerical factor.\n", GetTime() - t); t = GetTime();
        }

        W_u = A.solve_with_factor(B); // W_u = A\B; but more efficiently.

        if (TIMING) 
        {
            printf("Check point 4: %f.\t Back Sub 1.\n", GetTime() - t); t = GetTime();
        }

        // project the weights to the probability simplex. 
        FW_u = W2F(W_u);
        FW_k = BC;

        W.slice_assign_value(unknown, W_u, 0); // W(unknown, :) = W_u;
                
        double e = 1 / (i + 1); 
        // many optimizers do not rely on the energy value e, only its gradient. 
        // in this case e can be an arbitary value to save time.

        bool last_iter = i == (NUM_ITER * (REPEAT + 1)); // it's not (i+1) here. 

        if (verbose > 2 || last_iter) 
        {

            Dense H_u = Quu.mul(FW_u) + Quk.mul(FW_k);
            Dense H_k = Qku.mul(FW_u) + Qkk.mul(FW_k);
            e = ((Dense::times(FW_u, H_u).reduce_sum(0) + Dense::times(FW_k, H_k).reduce_sum(0)).reduce_sum(1))(0, 0);

            printf("\nIter %04d: energy=%f \t", i, e);

            if (verbose > 3) 
            {
                printf("W_u.min()=");
                W_u.reduce_min().print();
                printf("FW_u.min()=");
                FW_u.reduce_min().print();
            }

            if (i < length_his) 
            {
                energy_his[i] = e;
                time_his[i] = GetTime() - start_time;
            }
            if (LOG_HISTORY)
            {
                FW.slice_assign_value(unknown, FW_u, 0);
                char fname[64];
                sprintf(fname, "W%04d.mtx", i);
                FW.write(OUTPUT + fname); // projected weights
                sprintf(fname, "UW%04d.mtx", i);
                W.write(OUTPUT + fname); // unprojected weights
            }
        }
        else 
        {
            printf("Iter %04d...\t", i);
        }
        
        if (verbose>4) 
        {
            Dense NW = W2F_n(W);
            double pou = (W - NW).norm() / W.norm();
            printf("Checking partition of unity: %f, expecting ~0.\n", pou);
        }

        if (TIMING) 
        {
            printf("Check point 5: %f.\n", GetTime() - t); t = GetTime();
        }

        // Calculate gradients. 

        switch (4) { // 4 is fastest

        case 0:

            GW = G * W; // old code

            break;

        case 1:

            // [Gxu, Gxk]
            // [Gyu, Gyk] [W_u] = 
            // [Gzu, Gzk] [W_k]
            GW = Dense::concatenate(Gxu * W_u + Gxk * BC, Gyu * W_u + Gyk * BC);
            if (dim == 3)
                GW = Dense::concatenate(GW, Gzu * W_u + Gzk * BC);

            break;

        case 2:

            compute_grads(V, F, grad, W, GW); // somehow slower than G * W...

            break;

        case 3:

            tf_compute_grads(V, F, W, GW);

            break;

        case 4:

            grad_tf.run(W, GW);

            break;
        default:
            ;
        }
 
        if (TIMING) 
        {
            printf("Check point 6: %f. \t Gradient computation. \n", GetTime() - t); t = GetTime();
        }

        Res_u = (Quu.mul(FW_u) + Quk.mul(FW_k))* 2.0;

        Dense dW = dFW2dW(Res_u, W_u); // since dFW2dW is a row-wise operation

        if (TIMING) 
        {
            printf("Check point 7: %f.\n", GetTime() - t); t = GetTime();
        }

        BR = A.solve_with_factor(dW);

        if (TIMING) 
        {
            printf("Check point 8: %f. \t Back Sub 2\n", GetTime() - t); t = GetTime();
        }

        // Calculate gradients for the second time

        switch (2) { // 2 is fastest

        case 0:

            PS = Gu.mul(BR); // old code

            break;

        case 1:

            PS = Dense::concatenate(Gxu.mul(BR), Gyu.mul(BR));
            if (dim == 3)
                PS = Dense::concatenate(PS, Gzu.mul(BR));

            break;

        case 2:

            static Dense BRa;
            if (i==0)
                BRa = Dense::Zeros(V.nrow(), BR.ncol());

            BRa.slice_assign_value(unknown, BR, 0);

            grad_tf.run(BRa, PS);

            break;

        default:
            ;
        }
        
        if (TIMING) 
        {
            printf("Check point 9: %f. \t Mat Multiplication 2.\n", GetTime() - t); t = GetTime();
        }

        gau = Dense::Zeros(f * mcdim, 1);

        for (int j = 0; j < W.ncol(); j++) 
        {
            symmetric_tensor_span_dot(&GW(0, j), &PS(0, j), gau_j.head(), f, dim);
            // gau = gau - gau_j;
            Dense::saxy(gau_j, gau, -1.0);
        }

        if (TIMING) 
        {
            printf("Check point 10: %f.\n", GetTime() - t); t = GetTime();
        }

        /* gat <- gau: back-prop grad */
        
        // s_pdapdt_lmul_fast(at_tmp, gau, gat, FA, dim, para);
        s_pdapdt_lmul(at, gau, gat, FA, dim, para); 
        
        if (TIMING) 
        {
            printf("Check point 11: %f.\n", GetTime() - t); t = GetTime();
        }

        i++;
        return e;
        // e may not be the function value, depends on parameters. 
    };

    Dense at_out;
    {
        using arma::vec;
        using arma::mat;
        vec at = dense_array_to_arma_vec(at0);

        printf("Solver: %s\n", SOLVER.c_str());

        if (SOLVER==std::string("adamd")) 
        {
            /* Adam - based optim */

            optim::algo_settings_t settings;
            settings.gd_method = 6;
            settings.gd_settings.step_size = STEP_SIZE;
            settings.iter_max = NUM_ITER;

            std::function<double(const arma::vec&, arma::vec*, void*)> fn 
                = [&](const arma::vec& at, arma::vec* pgat, void* opt_data) {
                double v = fun(at, *pgat);
                return v;
            };

            for (int j = 0; j < REPEAT; j++)
            {
                optim::gd2(at, fn, NULL, settings); // gd2 gets rid of an extra call of fn at the end. 

                settings.gd_settings.step_size /= 2.0;
                printf("\nlr=%g \n", settings.gd_settings.step_size);
            }

            optim::gd(at, fn, NULL, settings);
            // NUM_ITER * REPEAT + NUM_ITER: in total

        } 
        else if (SOLVER == std::string("adam")) 
        {

            optim::algo_settings_t settings;
            settings.gd_method = 6;
            settings.gd_settings.step_size = STEP_SIZE;
            settings.iter_max = NUM_ITER * (REPEAT+1);

            std::function<double(const arma::vec&, arma::vec*, void*)> fn
                = [&](const arma::vec& at, arma::vec* pgat, void* opt_data) {
                double v = fun(at, *pgat);
                return v;
            };

            optim::gd(at, fn, NULL, settings);

        } 
        else if (SOLVER == std::string("lbfgs")) 
        {

            optim::algo_settings_t settings;
            settings.iter_max = NUM_ITER * (REPEAT + 1);
            settings.lbfgs_par_M = LBFGS_M;

            std::function<double(const arma::vec&, arma::vec*, void*)> fn
                = [&](const arma::vec& at, arma::vec* pgat, void* opt_data) {
                double v = fun(at, *pgat);
                return v;
            };

            optim::lbfgs(at, fn, NULL, settings);

        }
        else 
        {
            std::cerr << "--solver not supported." << std::endl;
        }

        at_out = arma_vec_to_dense_array(at);
    }

    printf("numerical_factor + solving: %f seconds.\n", GetTime() - start_time);

    if (LOG_HISTORY) 
    {
        printf("Timing is not meaningful due to logging cost. \n");
    }

    FW.slice_assign_value(unknown, FW_u, 0); // FW(unknown, :) = FW_u;
    FW.write(OUTPUT+"W.mtx"); // projected weights
    W.write(OUTPUT+"UW.mtx"); // unprojected weights
    at_out.write(OUTPUT+"at.mtx");
    au.write(OUTPUT + "au.mtx");


    End(cm);

    free(I);
    free(J);

    FILE* filename = fopen((OUTPUT+"log.txt").c_str(), "wb");
    if (filename != NULL) 
    {
        for (int i = 0; i < length_his; i++) 
        {
            fprintf(filename, "%04d\t%g\t%g\n", i, energy_his[i], time_his[i]);
        }
        fclose(filename);
    }

    free(energy_his);

    return 0;
}

// #include "iter_lap_timing.cpp"