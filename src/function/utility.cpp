#ifdef _UTIL_HPP_

#include <cassert>

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

#endif
