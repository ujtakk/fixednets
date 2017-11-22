#ifndef _FULL_HPP_
#define _FULL_HPP_

#include "matrix.hpp"

template <typename T>
void full(Mat1D<T>& output, Mat1D<T>& input, Mat2D<T>& weight);

template <typename T>
void gemm(Mat2D<T>& output, Mat2D<T>& input, Mat2D<T>& weight);

#include "full.cpp"
#endif
