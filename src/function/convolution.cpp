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

  const int fea_h = in_h + 2*pad - fil_h + 1;
  const int fea_w = in_w + 2*pad - fil_w + 1;

  Mat3D<T> padded = zeros<T>(n_in, in_h+2*pad+stride-1, in_w+2*pad+stride-1);

#if 1
  auto tf_pad = [](auto fil_size, auto in_size, auto stride) {
    int pad_whole;
    if (in_size % stride == 0)
      pad_whole = fil_size - stride;
    else
      pad_whole = fil_size - (in_size % stride);

    int pad_side;
    if (pad_whole < 0)
      pad_side = 0;
    else
      pad_side = pad_whole / 2;

    return pad_side;
  };
  const int tf_pad_h = tf_pad(fil_h, in_h, stride);
  const int tf_pad_w = tf_pad(fil_w, in_w, stride);
#endif

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int m = 0; m < n_in; ++m) {
    for (int i = 0; i < in_h; ++i) {
      for (int j = 0; j < in_w; ++j) {
        // # chainer mode
        // padded[m][i+pad][j+pad] = input[m][i][j];
        // # tensorflow mode
        padded[m][i+tf_pad_h][j+tf_pad_w] = input[m][i][j];
      }
    }
  }

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_out; ++n) {
    for (int m = 0; m < n_in; ++m) {
      for (int i = 0; i < fea_h; i+=stride) {
        for (int j = 0; j < fea_w; j+=stride) {
          T acc = 0;
          for (int k = 0; k < fil_h; ++k)
            for (int l = 0; l < fil_w; ++l)
              acc += mlt(weight[n][m][k][l], padded[m][i+k][j+l]);
          output[n][i/stride][j/stride] += acc;
        }
      }
    }
  }

  // for (int n = 0; n < n_out; ++n) {
  //   for (int i = 0; i < fea_h; i+=stride) {
  //     for (int j = 0; j < fea_w; j+=stride) {
  //       std::cout << output[n][i/stride][j/stride] << std::endl;
  //     }
  //   }
  // }
  // exit(0);
}

#endif
