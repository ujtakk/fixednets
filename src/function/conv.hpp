#ifndef _CONV_HPP_
#define _CONV_HPP_

#include "matrix.hpp"

template <typename T>
void conv_plus_pad(Mat3D<T>& input, Mat4D<T>& weight, Mat3D<T>& output,
                   int stride, int pad);

template <typename T>
Mat3D<T> conv_plus_pad(Mat3D<T>& input, Mat4D<T>& weight, int stride, int pad);

#include "conv.cpp"
#endif
