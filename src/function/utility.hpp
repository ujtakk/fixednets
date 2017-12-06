#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include "types.hpp"
#include "matrix.hpp"

template <typename F, typename T>
auto apply(F f, T v) -> T;
template <typename F, typename T>
auto apply(F f, std::vector<T> v) -> std::vector<T>;

auto to_fixed(float x) -> fixed;
auto to_float(fixed x) -> float;

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

inline float mlt(float a, float b);
inline fixed mlt(fixed a, fixed b);

inline float dvd(float a, float b);
inline fixed dvd(fixed a, fixed b);

template <typename T>
void bias(Mat1D<T>& output, Mat1D<T>& input, Mat1D<T>& bias);

template <typename T>
void bias(Mat3D<T>& output, Mat3D<T>& input, Mat1D<T>& bias);

#include "utility.cpp"
#endif
