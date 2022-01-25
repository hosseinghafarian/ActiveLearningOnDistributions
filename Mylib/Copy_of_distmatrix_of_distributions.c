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
//void   print_vec(double *p, double *q, mwSize d, double sum){
//     mwSize i;
// 	mexPrintf("\n For Vector [ ");
//     for(i=0;i<d;i++)
//         mexPrintf("%f,",p[i]);
// 	mexPrintf("]'\n with Vector [");
// 	for (i = 0; i<d; i++)
// 		mexPrintf("%f,", q[i]);
// 	mexPrintf("]' Kernel is %f", sum);
//}
//void   print_vec2(double *p, mwSize d){
//     mwSize i;
// 	mexPrintf("\n Vector [ ");
//     for(i=0;i<d;i++)
//         mexPrintf("%f,",p[i]);
// 	mexPrintf("]");
//}
// void print_matrix(double *p, mwSize d, mwSize n){
//     mwSize i,j;
//     mexPrintf("[");
//     for(i=0;i<d;i++){
//         for(j=0;j<n;j++)
//             mexPrintf(",%3.2f",p[i+j*n]);
//         mexPrintf(";");
//     }
//     mexPrintf("]");
//}
double rbf_kernel_of_dist(double gamma, mwSize d, mwSize mA, mwSize mB, double *A, double *B){
    mwSize i, j;
    double sum, value;		
    sum = 0;
    for(i = 0; i<mA; i++)
        for(j = 0 ; j<mB;j++){
		   value = rbf_kernel_compute(gamma, d, &A[i*d], &B[j*d]);
           //print_vec(&A[i*d], &A[j*d], d, value);
		   sum += value;
        }
    return sum;
}
double rbf_kernel_self_ofdist(double gamma, mwSize d, mwSize mA, double *A) {
	mwSize i, j;
	double sum, value,sumrow;
	sum = 0;
	for (i = 0; i < mA; i++) {
        sumrow = 0;
		for (j = i + 1; j < mA; j++) {
			value = rbf_kernel_compute(gamma, d, &A[i*d], &A[j*d]);
			sumrow += value;
            //print_vec(&A[i*d], &A[j*d], d, value);
		}
        sum += 2*sumrow;
        value = rbf_kernel_compute(gamma, d, &A[i*d], &A[i*d]);
		sum += value;
        //print_vec(&A[i*d], &A[i*d], d, value);
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
void distance_matrix( mwSize d, mwSize nA, double *A, double *F_A, mwSize midx_A, double *idx_dF_A, 
                                mwSize nB, double *B, double *F_B, mwSize midx_B, double*idx_dF_B, double gamma, double* dm ){
    
    const double tol = 1e-12;
    double *X_i, *X_j,value;
    mwSize i,j, stidx_A, stidx_B, enidx_A, enidx_B, nX_i, nX_j;
    stidx_A = 0;
    for(i=0; i < midx_A;i++){
       enidx_A = idx_dF_A[i];
       //print_vec2(idx_dF_A, midx_A);
       //mexPrintf("\n enidx_A:%d\n", enidx_A);
       X_i = &A[stidx_A*d]; nX_i = enidx_A-stidx_A;
       //mexPrintf("\n matrix:");
       //print_matrix(X_i, d, nX_i);
       stidx_B = 0;
       for(j =0; j<midx_B;j++){
           enidx_B = idx_dF_B[j];
           X_j = &B[stidx_B*d]; nX_j = enidx_B-stidx_B;
           //mexPrintf("\n with matrix:");
           //print_matrix(X_j, d, nX_j);
           value = normdiff_of_dist(gamma, d, nX_i, nX_j, X_i, X_j);
           if (value <tol){
               value = 0;
           }
           //mexPrintf("\n i:%d,j:%d, midx_A:%d, value:%4.3f",i,j,midx_A, value);
           dm[i*midx_A+j] = value;
           stidx_B = enidx_B;
       }
       stidx_A = enidx_A;
    }
}
void validate_inputs(mwSize *d, int*nA, int*nB, int nlhs, mxArray *plhs[],
                                                    int nrhs, const mxArray *prhs[] ) { 
    int mA, mB;
    /* check for proper number of arguments */
    if(nrhs!=7) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Seven inputs required.");
    }
//    if(nlhs!=1) {
//        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
//    }
     
    /* make sure the first input argument is type double */
    if( !mxIsDouble(prhs[0]) || 
         mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    mA  = mxGetM(prhs[0]);
    *nA  = mxGetN(prhs[0]);
    
    /* make sure the second input argument is type double */
    if( (!mxIsNumeric(prhs[1]) && mxIsDouble(prhs[1])) || mxGetN(prhs[1])!= *nA || mxGetM(prhs[1])!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Second argument must be a integer row vector the same size as columns of the first.");
    }
    if( !mxIsNumeric(prhs[2]) && mxIsDouble(prhs[2])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Third argument must be a row vector of integers");
    }
    
    /* make sure the first input argument is type double */
    if( !mxIsDouble(prhs[3]) || 
         mxIsComplex(prhs[3])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    mB  = mxGetM(prhs[3]);
    *nB  = mxGetN(prhs[3]);
    /*check that both of the first and second argument are in the same size*/
    if(mA != mB) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","First and Second Input must be a of the same size");
    }
    *d = mA;
    
    /* make sure the second input argument is type double */
    if( (!mxIsNumeric(prhs[4]) && mxIsDouble(prhs[4])) || mxGetN(prhs[4])!= *nB || mxGetM(prhs[4])!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Second argument must be a integer row vector the same size as columns of the first.");
    }
    if( !mxIsNumeric(prhs[5]) && mxIsDouble(prhs[5])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Third argument must be a row vector of integers");
    }
    if (!mxIsDouble(prhs[6])){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","gamma must be a double");
    }
}
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double gamma;              /* input scalar */
    double *A, *F_A, *idx_dF_A;               /* 1xN input matrix */
    double *B, *F_B, *idx_dF_B;               /* 1xN input matrix */
    size_t nrows, nA, mA, nB, mB, d, midx_A, midx_B;                   /* size of matrix */
    double *outMatrix;              /* output matrix */
    
    validate_inputs(&d,&nA,&nB,nlhs,plhs,nrhs,prhs );
     
    /* get the value of the scalar input  */
    gamma = mxGetScalar(prhs[6]);

    /* create a pointer to the real data in the input matrix  */
    A = mxGetPr(prhs[0]);
    F_A = mxGetPr(prhs[1]);
    //print_vec2(F_A, nA);
    idx_dF_A = mxGetPr(prhs[2]);
    midx_A   = mxGetN(prhs[2]);
    //print_vec2(idx_dF_A, midx_A);
    B = mxGetPr(prhs[3]);
    F_B = mxGetPr(prhs[4]);
    //print_vec2(F_B, nB);
    idx_dF_B = mxGetPr(prhs[5]);
    midx_B   = mxGetN(prhs[5]);
    //print_vec2(idx_dF_B, midx_B);
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(midx_A, midx_B, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);
    //mexPrintf("midx_A:%d\n",midx_A);
    distance_matrix(d, nA, A, F_A, midx_A, idx_dF_A, nB, B, F_B, midx_B, idx_dF_B, gamma, outMatrix );
    /* call the computational routine */
    //outMatrix[0] = normdiff_of_dist(gamma, d, nA, nB, A, B);
}