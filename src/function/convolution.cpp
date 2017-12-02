#ifdef _CONV_HPP_

#include "function.hpp"

template <typename T>
void conv_plus_pad(Mat3D<T>& output, Mat3D<T>& input, Mat4D<T>& weight,
                   int stride, int pad)
{
  const int n_out = weight.size();
  const int n_in = weight[0].size();

  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  const int fil_h = weight[0][0].size();
  const int fil_w = weight[0][0][0].size();

  const int fea_h = in_h - fil_h + 2*pad;
  const int fea_w = in_w - fil_w + 2*pad;

  // TODO: add static_assert

  Mat3D<T> padded = zeros<T>(n_in, in_h+2*pad+stride-1, in_w+2*pad+stride-1);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int m = 0; m < n_in; ++m)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        padded[m][i+pad][j+pad] = input[m][i][j];

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_out; ++n) {
    for (int m = 0; m < n_in; ++m) {
      for (int i = 0; i < fea_h; i+=stride) {
        for (int j = 0; j < fea_w; j+=stride) {
          for (int k = 0; k < fil_h; ++k)
            for (int l = 0; l < fil_w; ++l)
              output[n][i/stride][j/stride] +=
                mul(padded[m][i+k][j+l], weight[n][m][k][l]);
        }
      }
    }
  }
}

#endif
