#ifdef _UTIL_HPP_

#include <cassert>
#include <random>

#include "types.hpp"

inline float mul(float a, float b)
{
  return a * b;
}

template <typename BaseT>
inline BaseT mul(BaseT a, BaseT b)
{
  using MultT = int64_t;
  MultT c = a * b;

  if (c >= 0)
    return c / Q_OFFSET<BaseT>;
  else
    return c / Q_OFFSET<BaseT> - 1;
}

template <typename T>
void bias(Mat1D<T>& output, Mat1D<T>& input, Mat1D<T>& bias)
{
  const int n_in = input.size();

  for (int n = 0; n < n_in; ++n)
    output[n] = input[n] + bias[n];
}

template <typename T>
void bias(Mat3D<T>& output, Mat3D<T>& input, Mat1D<T>& bias)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  for (int n = 0; n < n_in; ++n)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        output[n][i][j] = input[n][i][j] + bias[n];
}

template <typename T>
void flatten(Mat1D<T>& output, Mat3D<T>& input)
{
  const int mdep = input.size();
  const int mhei = input[0].size();
  const int mwid = input[0][0].size();

  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        output[i*mhei*mwid+j*mwid+k] = input[i][j][k];
}

template <typename T>
void reshape(Mat3D<T>& output, Mat1D<T>& input)
{
  const int mdep = output.size();
  const int mhei = output[0].size();
  const int mwid = output[0][0].size();

  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        output[i][j][k] = input[i*mhei*mwid+j*mwid+k];
}

template <typename T>
void concat(Mat3D<T>& c, Mat3D<T>& a, Mat3D<T>& b)
{
  const int n_a = a.size();
  const int n_b = b.size();
  const int n_c = c.size();

  assert(n_c == n_b + n_a);

  const int im_h = c[0].size();
  const int im_w = c[0][0].size();

  for (int i = 0; i < n_a; ++i)
    for (int j = 0; j < im_h; ++j)
      for (int k = 0; k < im_w; ++k)
        c[i][j][k] = a[i][j][k];

  for (int i = 0; i < n_b; ++i)
    for (int j = 0; j < im_h; ++j)
      for (int k = 0; k < im_w; ++k)
        c[i+n_a][j][k] = b[i][j][k];
}

#endif
