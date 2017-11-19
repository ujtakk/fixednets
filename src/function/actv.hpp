#ifndef _ACTV_HPP_
#define _ACTV_HPP_

#include "matrix.hpp"

template <typename T>
void relu(Mat3D<T>& input, Mat3D<T>& output);

template <typename T>
void relu(Mat1D<T>& input, Mat1D<T>& output);

int softmax(Mat1D<double>& output, int len);

#include "actv.cpp"
#endif
