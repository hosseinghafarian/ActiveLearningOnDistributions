/*
 * rand.c
 *
 * Code generation for function 'rand'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "test_parfor.h"
#include "rand.h"

/* Function Definitions */
void b_rand(real_T r[2560])
{
  emlrtRandu(r, 2560);
}

/* End of code generation (rand.c) */
