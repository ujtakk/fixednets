#ifdef _UTIL_HPP_

#include <cassert>

#include "types.hpp"

template <typename Func, typename T>
auto apply(Func f, T v) -> T
{
  return f(v);
}

template <typename Func, typename T>
auto apply(Func f, std::vector<T> v) -> std::vector<T>
{
  const int len = v.size();
  std::vector<T> fv(len);

  for (int i = 0; i < len; ++i)
    fv[i] = apply(f, v[i]);

  return fv;
}

auto to_fixed(float x) -> fixed
{
  return static_cast<fixed>(rint(x * Q_OFFSET<float>));
}

auto to_float(fixed x) -> float
{
  return static_cast<float>(x) / Q_OFFSET<float>;
}

template <typename T>
inline float float_of_T(T& x)
{
  // return static_cast<float>(x) / Q_OFFSET<float>;
  return to_float(x);
}

#include <iostream>
inline float float_of_T(float& x)
{
  return x;
}

// Mats
// {{{
template <typename T>
inline T T_of_float(float& x)
{
  // return static_cast<T>(rint(x * Q_OFFSET<float>));
  return to_fixed(x);
}

inline float T_of_float(float& x)
{
  return x;
}

template <typename T>
inline Mat1D<T> T_of_float(Mat1D<float>& x)
{
  const int size0 = x.size();

  auto y = zeros<T>(size0);

  for (int i = 0; i < size0; ++i)
    y[i] = T_of_float(x[i]);

  return y;
}

template <typename T>
inline Mat2D<T> T_of_float(Mat2D<float>& x)
{
  const int size0 = x.size();
  const int size1 = x[0].size();

  auto y = zeros<T>(size0, size1);

  for (int i = 0; i < size0; ++i)
    for (int j = 0; j < size1; ++j)
      y[i][j] = T_of_float(x[i][j]);

  return y;
}

template <typename T>
inline Mat3D<T> T_of_float(Mat3D<float>& x)
{
  const int size0 = x.size();
  const int size1 = x[0].size();
  const int size2 = x[0][0].size();

  auto y = zeros<T>(size0, size1, size2);

  for (int i = 0; i < size0; ++i)
    for (int j = 0; j < size1; ++j)
      for (int k = 0; k < size2; ++k)
        y[i][j][k] = T_of_float(x[i][j][k]);

  return y;
}

template <typename T>
inline Mat4D<T> T_of_float(Mat4D<float>& x)
{
  const int size0 = x.size();
  const int size1 = x[0].size();
  const int size2 = x[0][0].size();
  const int size3 = x[0][0][0].size();

  auto y = zeros<fixed>(size0, size1, size2, size3);

  for (int i = 0; i < size0; ++i)
    for (int j = 0; j < size1; ++j)
      for (int k = 0; k < size2; ++k)
        for (int l = 0; l < size3; ++l)
          y[i][j][k][l] = T_of_float(x[i][j][k][l]);

  return y;
}

template <typename T>
inline Mat1D<float> float_of_T(Mat1D<T>& x)
{
  const int size0 = x.size();

  auto y = zeros<float>(size0);

  for (int i = 0; i < size0; ++i)
    y[i] = float_of_T(x[i]);

  return y;
}

template <typename T>
inline Mat2D<float> float_of_T(Mat2D<T>& x)
{
  const int size0 = x.size();
  const int size1 = x[0].size();

  auto y = zeros<float>(size0, size1);

  for (int i = 0; i < size0; ++i)
    for (int j = 0; j < size1; ++j)
      y[i][j] = float_of_T(x[i][j]);

  return y;
}

template <typename T>
inline Mat3D<float> float_of_T(Mat3D<T>& x)
{
  const int size0 = x.size();
  const int size1 = x[0].size();
  const int size2 = x[0][0].size();

  auto y = zeros<float>(size0, size1, size2);

  for (int i = 0; i < size0; ++i)
    for (int j = 0; j < size1; ++j)
      for (int k = 0; k < size2; ++k)
        y[i][j][k] = float_of_T(x[i][j][k]);

  return y;
}

template <typename T>
inline Mat4D<float> float_of_T(Mat4D<T>& x)
{
  const int size0 = x.size();
  const int size1 = x[0].size();
  const int size2 = x[0][0].size();
  const int size3 = x[0][0][0].size();

  auto y = zeros<float>(size0, size1, size2, size3);

  for (int i = 0; i < size0; ++i)
    for (int j = 0; j < size1; ++j)
      for (int k = 0; k < size2; ++k)
        for (int l = 0; l < size3; ++l)
          y[i][j][k][l] = float_of_T(x[i][j][k][l]);

  return y;
}
// }}}

inline float mlt(float a, float b)
{
  return a * b;
}

inline fixed mlt(fixed a, fixed b)
{
  // int64_t c = (a * b) >> Q_BITS;
  // int64_t c = (a * b) / Q_OFFSET<float>;
  int64_t c = a * b;

  // if (c < 0)
  //   return c - 1;
  // else
  //   return c;
  if (c < 0)
    return (c >> Q_BITS) - 1;
  else
    return (c >> Q_BITS);
  // if (c >= 0)
  //   return c / Q_OFFSET<fixed>;
  // else
  //   return c / Q_OFFSET<fixed> - 1;
}

inline float dvd(float a, float b)
{
  return a / b;
}

inline fixed dvd(fixed a, fixed b)
{
  int64_t c = (a << Q_BITS) / b;
  // int64_t c = (a * Q_OFFSET<float>) / b;

  return c;
}

template <typename T>
void bias(Mat1D<T>& output, Mat1D<T>& input, Mat1D<T>& bias)
{
  const int n_in = input.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_in; ++n)
    output[n] = input[n] + bias[n];
}

template <typename T>
void bias(Mat3D<T>& output, Mat3D<T>& input, Mat1D<T>& bias)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_in; ++n)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        output[n][i][j] = input[n][i][j] + bias[n];
}

#endif
