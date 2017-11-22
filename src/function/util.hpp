#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include "matrix.hpp"

inline float mul(float a, float b);

template <typename BaseT>
inline BaseT mul(BaseT a, BaseT b);

template <typename T>
void bias(Mat1D<T>& input, Mat1D<T>& bias, Mat1D<T>& output);

template <typename T>
void bias(Mat3D<T>& input, Mat1D<T>& bias, Mat3D<T>& output);

template <typename T>
Mat1D<T> bias(Mat1D<T>& input, Mat1D<T>& output);

template <typename T>
Mat3D<T> bias(Mat3D<T>& input, Mat1D<T>& output);

int approx(int value, int bias, double prob);

template <typename T>
void flatten(Mat3D<T>& matrix, Mat1D<T>& array);

template <typename T>
void reshape(Mat1D<T>& array, Mat3D<T>& matrix);

template <typename T>
void concat(Mat3D<T>& a, Mat3D<T>& b, Mat3D<T>& c);

template <typename T>
Mat1D<T> flatten(Mat3D<T>& matrix);

template <typename T>
Mat3D<T> reshape(Mat1D<T>& array, int shape[3]);

template <typename T>
void concat(Mat3D<T>& a, Mat3D<T>& b, Mat3D<T>& c);

template <typename T>
Mat3D<T> concat(Mat3D<T>& a, Mat3D<T>& b);

#include "util.cpp"
#endif
