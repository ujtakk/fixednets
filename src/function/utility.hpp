#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include "matrix.hpp"

template <typename T>
inline T T_of_float(float& x);

inline float T_of_float(float& x);

template <typename T>
inline Mat1D<T> T_of_float(Mat1D<float>& x);

template <typename T>
inline Mat2D<T> T_of_float(Mat2D<float>& x);

template <typename T>
inline Mat3D<T> T_of_float(Mat3D<float>& x);

template <typename T>
inline Mat4D<T> T_of_float(Mat4D<float>& x);

template <typename T>
inline float float_of_T(T& x);

inline float float_of_T(float& x);

template <typename T>
inline Mat1D<float> float_of_T(Mat1D<T>& x);

template <typename T>
inline Mat2D<float> float_of_T(Mat2D<T>& x);

template <typename T>
inline Mat3D<float> float_of_T(Mat3D<T>& x);

template <typename T>
inline Mat4D<float> float_of_T(Mat4D<T>& x);

inline float mul(float a, float b);

template <typename BaseT>
inline BaseT mul(BaseT a, BaseT b);

template <typename T>
void bias(Mat1D<T>& output, Mat1D<T>& input, Mat1D<T>& bias);

template <typename T>
void bias(Mat3D<T>& output, Mat3D<T>& input, Mat1D<T>& bias);

#include "utility.cpp"
#endif
