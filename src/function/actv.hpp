#ifndef _ACTV_HPP_
#define _ACTV_HPP_

#include "matrix.hpp"

template <typename T>
void relu(Mat3D<T>& input, Mat3D<T>& output);

template <typename T>
void relu(Mat1D<T>& input, Mat1D<T>& output);

void softmax(Mat1D<float>& input, Mat1D<float>& output);

template <typename T>
Mat3D<T> relu(Mat3D<T>& input);

template <typename T>
Mat1D<T> relu(Mat1D<T>& input);

Mat1D<float> softmax(Mat1D<float>& input);

#include "actv.cpp"
#endif
