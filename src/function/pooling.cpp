#ifdef _POOL_HPP_

#include <limits>

template <typename T>
void pool_max(Mat3D<T>& output, Mat3D<T>& input,
              int fil_h, int fil_w, int stride, int pad)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  const int fea_h = in_h - fil_h + stride-1 + 2*pad;
  const int fea_w = in_w - fil_w + stride-1 + 2*pad;

  Mat3D<T> padded = zeros<T>(n_in, in_h+2*pad+stride-1, in_w+2*pad+stride-1);
  for (int m = 0; m < n_in; ++m)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        padded[m][i+pad][j+pad] = input[m][i][j];

  Mat1D<T> max(n_in, std::numeric_limits<T>::min());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_in; ++n) {
    for (int i = 0; i < fea_h; i+=stride) {
      for (int j = 0; j < fea_w; j+=stride) {
        for (int k = 0; k < fil_h; ++k) {
          for (int l = 0; l < fil_w; ++l) {
            if (padded[n][i+k][j+l] > max[n])
              max[n] = padded[n][i+k][j+l];
          }
        }
        output[n][i/stride][j/stride] = max[n];
        max[n] = std::numeric_limits<T>::min();
      }
    }
  }
}

#endif
