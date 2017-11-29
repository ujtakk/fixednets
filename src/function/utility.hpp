#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include "matrix.hpp"

inline float mul(float a, float b);

template <typename BaseT>
inline BaseT mul(BaseT a, BaseT b);

template <typename T>
void bias(Mat1D<T>& output, Mat1D<T>& input, Mat1D<T>& bias);

template <typename T>
void bias(Mat3D<T>& output, Mat3D<T>& input, Mat1D<T>& bias);

#include "utility.cpp"
#endif
