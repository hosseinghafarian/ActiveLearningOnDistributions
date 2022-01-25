/*==========================================================
 * norm_diff_of_distribution.c - 
 *
 * Computes distance between two emprical distrbutions using 
 * Embdedded Kernel  
 * P_1 : A distribution samples in the form d x nA matrix (A)
 * P_2 : A distribution samples in the form d x nB matrix (B)
 * gamma: gamma of RBF kernel for embedding kernel
 * and outputs a 1x1 matrix (outMatrix)
 *
 * The calling syntax is:
 *
 *		outMatrix = norm_diff_of_distrbution(A,B, gamma)
 *
 * This is a MEX-file for MATLAB.
 * Written by Hossein Ghafarian. 
 *
 *========================================================*/
#include "math.h"
#include "mex.h"

/* The computational routine */
double rbf_kernel_compute(double gamma, mwSize d, double *y, double *z)
{
    mwSize i;
    double sum, retval, diff;
    /* multiply each element y by x */
    sum = 0;
//	mexPrintf("^^^^");
    for (i=0; i<d; i++) {
        diff = z[i]- y[i];
//		mexPrintf("===%f-%f=%f===", z[i], y[i], diff);
        sum += diff*diff;
        
    }
    retval = exp(-0.5*gamma*sum);
    return retval;
}
void   print_vec(double *p, double *q, mwSize d, double sum){
    mwSize i;
	mexPrintf("\n For Vector [ ");
    for(i=0;i<d;i++)
        mexPrintf("%f,",p[i]);
	mexPrintf("]'\n with Vector [");
	for (i = 0; i<d; i++)
		mexPrintf("%f,", q[i]);
	mexPrintf("]' Kernel is %f", sum);
}
double rbf_kernel_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B){
    mwSize i, j;
    
    double sum, value;		
    sum = 0;
    for(i = 0; i<mA; i++)
        for(j = 0 ; j<mB;j++){
		   value = rbf_kernel_compute(gamma, d, &A[i*d], &B[j*d]);
		   sum += value;
//		   print_vec(&A[i*d], &B[j*d], d, value);
        }
    return sum;
}
double rbf_kernel_self_ofdist(double gamma, mwSize d, mwSize mA, double *A) {
	mwSize i, j;
	double sum, value;
	sum = 0;
	for (i = 0; i < mA; i++) {
		for (j = i + 1; j < mA; j++) {
			value = rbf_kernel_compute(gamma, d, &A[i*d], &A[j*d]);
			sum += value;
//			print_vec(&A[i*d], &A[j*d], d, value);
		}
		sum += rbf_kernel_compute(gamma, d, &A[i*d], &A[i*d]);
	}
	return sum;
}
double normdiff_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B){
    double sAA, sBB, sAB, result;
    sAA = rbf_kernel_self_ofdist(gamma, d, mA, A);
    sAB = rbf_kernel_of_dist(gamma, d, mA, mB, A, B); 
    sBB = rbf_kernel_self_ofdist(gamma, d, mB, B);
    result = sAA/(mA*mA) + sBB/(mB*mB) - 2*sAB/(mA*mB);
    return result;
}
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double gamma;              /* input scalar */
    double *A;               /* 1xN input matrix */
    double *B;               /* 1xN input matrix */
    size_t nrows, nA, mA, nB, mB, d;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
     
    /* make sure the first input argument is type double */
    if( !mxIsDouble(prhs[0]) || 
         mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[1]) || 
         mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    /* make sure the third input argument is scalar */
    if( !mxIsDouble(prhs[2]) || 
         mxIsComplex(prhs[2]) ||
         mxGetNumberOfElements(prhs[2])!=1 ) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notScalar","Input multiplier must be a scalar.");
    }
    mA  = mxGetM(prhs[0]);
    nA  = mxGetN(prhs[0]);
    mB  = mxGetM(prhs[1]);
    nB  = mxGetN(prhs[1]);
    
    /*check that both of the first and second argument are in the same size*/
    if(mA != mB) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","First and Second Input must be a of the same size");
    }
    d = mA;
    /* get the value of the scalar input  */
    gamma = mxGetScalar(prhs[2]);

    /* create a pointer to the real data in the input matrix  */
    A = mxGetPr(prhs[0]);
    B = mxGetPr(prhs[1]);
        /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    /* call the computational routine */
    outMatrix[0] = normdiff_of_dist(gamma, d, nA, nB, A, B);
}