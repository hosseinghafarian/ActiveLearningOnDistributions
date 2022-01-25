% This make.m is used under Windows

% add -largeArrayDims on 64-bit machines

mex -O -c mymatrix_lib.cpp -largeArrayDims
mex -O matrixInverse.cpp mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims
mex -O matrixMultiply.cpp mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims


mex -g -c kernel_lib.cpp -largeArrayDims
mex -g distmatrix_of_distributions.cpp kernel_lib.obj mymatrix_lib.obj  libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g distmatrix_of_distributions_sameset.cpp kernel_lib.obj  mymatrix_lib.obj  libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g rbf_emp_mex.cpp kernel_lib.obj  mymatrix_lib.obj  libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g KME_fval.cpp kernel_lib.obj   mymatrix_lib.obj  libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g KME_fval_all.cpp kernel_lib.obj   mymatrix_lib.obj  libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g KME_fval_all_dist.cpp kernel_lib.obj   mymatrix_lib.obj  libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g KME_fval_all_dist_test.cpp kernel_lib.obj   mymatrix_lib.obj  libmwlapack.lib libmwblas.lib -largeArrayDims


mex -g compute_R.cpp kernel_lib.obj mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g compute_R_inv_forall.cpp kernel_lib.obj mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g compute_KBayes.cpp kernel_lib.obj mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g normw_estimate.cpp kernel_lib.obj mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims
mex -g solveAx_b.cpp kernel_lib.obj mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims