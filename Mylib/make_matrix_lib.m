% This make.m is used under Windows

% add -largeArrayDims on 64-bit machines

mex -O -c mymatrix_lib.cpp -largeArrayDims
mex -O matrixInverse.cpp mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims
mex -O matrixMultiply.cpp mymatrix_lib.obj libmwlapack.lib libmwblas.lib -largeArrayDims