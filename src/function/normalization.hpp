#ifndef _NORM_HPP_
#define _NORM_HPP_

#include "matrix.hpp"

template <typename T>
void norm_batch(Mat1D<T>& output, Mat1D<T>& input, T gamma, T beta, T eps, T mean, T std);

template <typename T>
void norm_batch(Mat3D<T>& output, Mat3D<T>& input, T gamma, T beta, T eps, T mean, T std);

#include "normalization.cpp"
#endif
