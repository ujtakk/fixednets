#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include "matrix.hpp"

void add_bias(Mat2D<int> &input, int bias, int ihei, int iwid);

double mean_1d(Mat1D<double> vec);

int approx(int value, int bias, double prob);

template <typename T>
void flatten(Mat3D<T> &matrix, Mat1D<T> &array);

template <typename T>
void flatten(Mat3D<T> &matrix, Mat1D<T> &array,
             const int mdep, const int mhei, const int mwid);

template <typename T>
void reshape(Mat1D<T> &array, Mat3D<T> &matrix);

template <typename T>
void reshape(Mat1D<T> &array, Mat3D<T> &matrix,
             const int mdep, const int mhei, const int mwid);

#include "util.cpp"
#endif
