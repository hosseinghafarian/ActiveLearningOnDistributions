/*
 * fft.c
 *
 * Code generation for function 'fft'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "test_parfor.h"
#include "fft.h"

/* Function Definitions */
creal_T fft(real_T x)
{
  creal_T y;
  y.re = x;
  y.im = 0.0;
  return y;
}

/* End of code generation (fft.c) */
