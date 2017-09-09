#ifndef _ACTV_HPP_
#define _ACTV_HPP_

#include "matrix.hpp"

void activate(Mat2D<int> &input, const int ihei, const int iwid);

void activate_1d(Mat1D<int> &input, const int ilen);

int softmax(Mat1D<double> &output, int len);

#include "actv.cpp"
#endif
