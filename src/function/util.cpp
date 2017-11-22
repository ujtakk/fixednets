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
void bias(Mat1D<T>& input, Mat1D<T>& bias, Mat1D<T>& output)
{
  const int n_in = input.size();

  for (int n = 0; n < n_in; ++n)
    output[n] = input[n] + bias[n];
}

template <typename T>
void bias(Mat3D<T>& input, Mat1D<T>& bias, Mat3D<T>& output)
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
Mat1D<T> bias(Mat1D<T>& input, Mat1D<T>& bias)
{
  const int n_in = input.size();
  auto output = zeros<T>(n_in);

  for (int n = 0; n < n_in; ++n)
    output[n] = input[n] + bias[n];

  return output;
}

template <typename T>
Mat3D<T> bias(Mat3D<T>& input, Mat1D<T>& bias)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();
  auto output = zeros<T>(n_in, in_h, in_w);

  for (int n = 0; n < n_in; ++n)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        output[n][i][j] = input[n][i][j] + bias[n];
}

template <typename T>
void flatten(Mat3D<T>& matrix, Mat1D<T>& array)
{
  const int mdep = matrix.size();
  const int mhei = matrix[0].size();
  const int mwid = matrix[0][0].size();

  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        array[i*mhei*mwid+j*mwid+k] = matrix[i][j][k];
}

template <typename T>
void reshape(Mat1D<T>& array, Mat3D<T>& matrix)
{
  const int mdep = matrix.size();
  const int mhei = matrix[0].size();
  const int mwid = matrix[0][0].size();

  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        matrix[i][j][k] = array[i*mhei*mwid+j*mwid+k];
}

template <typename T>
Mat1D<T> flatten(Mat3D<T>& input)
{
  const int mdep = input.size();
  const int mhei = input[0].size();
  const int mwid = input[0][0].size();
  auto output = zeros<T>(mdep * mhei * mwid);

  int idx = 0;
  for (int i = 0; i < mdep; ++i) {
    for (int j = 0; j < mhei; ++j) {
      for (int k = 0; k < mwid; k++) {
        output[idx] = input[i][j][k];
        ++idx;
      }
    }
  }

  return output;
}

template <typename T>
Mat3D<T> reshape(Mat1D<T>& input, int shape[3])
{
  const int mdep = shape[0];
  const int mhei = shape[1];
  const int mwid = shape[2];
  auto output = zeros<T>(mdep * mhei * mwid);

  int idx = 0;
  for (int i = 0; i < mdep; i++) {
    for (int j = 0; j < mhei; j++) {
      for (int k = 0; k < mwid; k++) {
        output[i][j][k] = input[idx];
        ++idx;
      }
    }
  }

  return output;
}

template <typename T>
void concat(Mat3D<T>& a, Mat3D<T>& b, Mat3D<T>& c)
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

template <typename T>
Mat3D<T> concat(Mat3D<T>& a, Mat3D<T>& b)
{
  const int n_a = a.size();
  const int n_b = b.size();
  const int im_h = a[0].size();
  const int im_w = a[0][0].size();
  auto c = zeros<T>(n_a + n_b, im_h, im_w);

  // assert(n_c == n_b + n_a);
  // assert(b[0].size() == im_h);
  // assert(b[0][0].size() == im_w);

  for (int i = 0; i < n_a; ++i)
    for (int j = 0; j < im_h; ++j)
      for (int k = 0; k < im_w; ++k)
        c[i][j][k] = a[i][j][k];

  for (int i = 0; i < n_b; ++i)
    for (int j = 0; j < im_h; ++j)
      for (int k = 0; k < im_w; ++k)
        c[i+n_a][j][k] = b[i][j][k];

  return c;
}

#endif
