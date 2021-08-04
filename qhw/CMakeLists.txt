option(USE_EIGEN "" OFF)
option(USE_LBFGSPP "" OFF)
option(USE_DEC_VERSION "" OFF)

if(USE_EIGEN)
    add_definitions(-DUSE_EIGEN)
endif(USE_EIGEN)

if(USE_LBFGSPP)
    add_definitions(-DUSE_LBFGSPP)
    add_definitions(-DUSE_EIGEN)
endif(USE_LBFGSPP)

if(USE_DEC_VERSION)
    add_definitions(-DUSE_DEC_VERSION)
endif(USE_DEC_VERSION)


# TODO: use find_package()

set(MKL_INCLUDE /usr/include/mkl/)  
# /usr/include/mkl if installed by apt-get

set(MKL_LINK_FLAGS -lmkl_lapack95_lp64 -lmkl_blas95_lp64 -lmkl_gf_lp64 -lmkl_sequential -lmkl_core)

set(SOURCE_SUITESPARSE_PATH /qhw/external/SuiteSparse-5.7.1/)
set(LIB_SUITESPARSE_PATH /qhw/external/SuiteSparse-5.7.1/)

set(ARMA_INCLUDE_PATH /usr/include/) 
# /usr/include/armadillo is the default place for apt-get.

set(TENSORFLOW_PATH /qhw/external/tensorflow/2.4.0/)


# configure_file(tf_graphs ${CMAKE_CURRENT_BINARY_DIR}/tf_graphs COPYONLY)
# configure_file: not for repos. 

file(COPY tf_graphs DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

add_compile_options(-Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-value)
######################################

#  file(GLOB LIBOPTIM_SRC xxx) does not work for remote debug. 

set(LIBOPTIM_SRC
    "../optim/src/zeros/broyden.cpp"
    "../optim/src/constrained/sumt.cpp"
    "../optim/src/unconstrained/bfgs.cpp"
    "../optim/src/unconstrained/gd.cpp"
    "../optim/src/unconstrained/lbfgs.cpp"
    "../optim/src/unconstrained/newton.cpp"
    "../optim/src/line_search/more_thuente.cpp"
)

add_executable(demo_optim2 demo_optim.cpp ${LIBOPTIM_SRC}) 

target_compile_features(demo_optim2 PRIVATE cxx_std_17)

set(LIBOPTIM_COMPILE_OPTIONS -Wall -fPIC -O3 -march=native -ffp-contract=fast -flto -DNDEBUG -DARMA_NO_DEBUG -fopenmp)
set(LIBOPTIM_COMPILE_OPTIONS_DEBUG -fopenmp)

set(LIBOPTIM_LINK_FLAGS    -llapack -lblas -fopenmp -lpthread -lm)
set(LIBOPTIM_INCLUDE_DIRS    ${ARMA_INCLUDE_PATH} ../optim/include ) # "${CMAKE_SOURCE_DIR}/optim/include"

target_compile_options(demo_optim2 PRIVATE ${LIBOPTIM_COMPILE_OPTIONS})

target_include_directories(demo_optim2 PRIVATE ${LIBOPTIM_INCLUDE_DIRS})

target_link_libraries(demo_optim2 PRIVATE  ${LIBOPTIM_LINK_FLAGS} )


################### demo_tensorflow ####################

add_executable(demo_tf demo_tf.cpp tf_wrapper.cpp)

set(TENSORFLOW_INCLUDE_DIRS     ${TENSORFLOW_PATH}/include)
set(TENSORFLOW_LINK_FLAGS       ${TENSORFLOW_PATH}/lib/libtensorflow.so)

target_include_directories(demo_tf PRIVATE ${TENSORFLOW_INCLUDE_DIRS})

target_link_libraries(demo_tf ${TENSORFLOW_LINK_FLAGS})

#################### prepare for quasi_harmonic ####################

set(CHOLMOD_PATH "${SOURCE_SUITESPARSE_PATH}/CHOLMOD/")

set(SUITESPARSE_INCLUDE     "${LIB_SUITESPARSE_PATH}/include"   "${CHOLMOD_PATH}"   "${CHOLMOD_PATH}/Include"   "${CHOLMOD_PATH}/MATLAB"    "${CHOLMOD_PATH}/Core") 
set(SUITESPARSE_LINK_FLAGS  "${LIB_SUITESPARSE_PATH}/lib/libsuitesparseconfig.so"   "${LIB_SUITESPARSE_PATH}/lib/libcholmod.so")

# should also add ${CHOLMOD_PATH}/Include here for cholmod_internal.h etc. 

set(EIGEN_INCLUDE /usr/include/eigen3/)
# no need to link libraries for eigen

set(LBFGSPP_INCLUDE "${CMAKE_SOURCE_DIR}/external/LBFGSpp/include")

set(QHW_SRC qhw.cpp  alap.cpp helper.cpp mkl_for_cholmod.cpp cholmod_l.cpp matrix.cpp cholmod_plus.cpp mesh.cpp "${LIBOPTIM_SRC}" tf_wrapper.cpp matrix_dense.cpp matrixIO.cpp)

#################### quasi_harmonic ####################

add_executable(qhw ${QHW_SRC})

set_target_properties(qhw PROPERTIES COMPILE_FLAGS "-fopenmp -DMKL_INT=long") 

target_include_directories(qhw PRIVATE ${MKL_INCLUDE}  ${SUITESPARSE_INCLUDE} ${LIBOPTIM_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})

target_compile_features(qhw PRIVATE cxx_std_17)

target_compile_options(qhw PRIVATE ${LIBOPTIM_COMPILE_OPTIONS})

target_link_libraries(qhw -lm -fopenmp -lpthread ${MKL_LINK_FLAGS} ${SUITESPARSE_LINK_FLAGS} ${TENSORFLOW_LINK_FLAGS})

