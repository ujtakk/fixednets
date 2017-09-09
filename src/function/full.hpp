#ifndef _FULL_HPP_
#define _FULL_HPP_

#include "matrix.hpp"

void full_connect(
  Mat1D<int> &input, Mat1D<int> &output,
  Mat2D<int> &weight, Mat1D<int> &bias,
  const int ilen, const int olen
);

#include "full.cpp"
#endif
