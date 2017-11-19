#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include "matrix.hpp"

template <typename BaseT>
BaseT mult_fixed(BaseT a, BaseT b);

template <typename T>
void bias(Mat1D<T>& input, Mat1D<T>& bias, Mat1D<T>& output);

template <typename T>
void bias(Mat3D<T>& input, Mat1D<T>& bias, Mat3D<T>& output);

int approx(int value, int bias, double prob);

template <typename T>
void flatten(Mat3D<T>& matrix, Mat1D<T>& array);

template <typename T>
void flatten(Mat3D<T>& matrix, Mat1D<T>& array,
             const int mdep, const int mhei, const int mwid);

template <typename T>
void reshape(Mat1D<T>& array, Mat3D<T>& matrix);

template <typename T>
void reshape(Mat1D<T>& array, Mat3D<T>& matrix,
             const int mdep, const int mhei, const int mwid);

#include "util.cpp"
#endif
