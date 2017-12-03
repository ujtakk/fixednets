#ifndef _ACTV_HPP_
#define _ACTV_HPP_

#include "matrix.hpp"

template <typename T>
void relu(Mat3D<T>& output, Mat3D<T>& input);

template <typename T>
void relu(Mat1D<T>& output, Mat1D<T>& input);

template <typename T>
void softmax(Mat1D<T>& output, Mat1D<T>& input);

void softmax(Mat1D<float>& output, Mat1D<float>& input);

#include "activation.cpp"
#endif
