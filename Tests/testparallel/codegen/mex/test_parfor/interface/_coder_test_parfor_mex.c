/*
 * _coder_test_parfor_mex.c
 *
 * Code generation for function '_coder_test_parfor_mex'
 *
 */

/* Include files */
#include "test_parfor.h"
#include "_coder_test_parfor_mex.h"
#include "test_parfor_terminate.h"
#include "_coder_test_parfor_api.h"
#include "test_parfor_initialize.h"
#include "test_parfor_data.h"

/* Function Declarations */
static void test_parfor_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs);

/* Function Definitions */
static void test_parfor_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs)
{
  const mxArray *outputs[1];
  int32_T b_nlhs;
  emlrtStack st = { NULL, NULL, NULL };

  st.tls = emlrtRootTLSGlobal;

  /* Check for proper number of arguments. */
  if (nrhs != 0) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 0, 4,
                        11, "test_parfor");
  }

  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 11,
                        "test_parfor");
  }

  /* Call the function. */
  test_parfor_api(outputs);

  /* Copy over outputs to the caller. */
  if (nlhs < 1) {
    b_nlhs = 1;
  } else {
    b_nlhs = nlhs;
  }

  emlrtReturnArrays(b_nlhs, plhs, outputs);

  /* Module termination. */
  test_parfor_terminate();
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  (void)prhs;

  /* Initialize the memory manager. */
  mexAtExit(test_parfor_atexit);

  /* Module initialization. */
  test_parfor_initialize();

  /* Dispatch the entry-point. */
  test_parfor_mexFunction(nlhs, plhs, nrhs);
}

/* End of code generation (_coder_test_parfor_mex.c) */
