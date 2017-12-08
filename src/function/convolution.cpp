#ifdef _CONV_HPP_

#include "function.hpp"

template <typename T>
static Mat3D<T> make_pad(Mat3D<T>& input, Mat4D<T>& weight,
                         int stride, int pad)
{
  const int n_in = weight[0].size();

  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  Mat3D<T> padded = zeros<T>(n_in, in_h+2*pad+stride-1, in_w+2*pad+stride-1);

#if 1
  const int fil_h = weight[0][0].size();
  const int fil_w = weight[0][0][0].size();

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
  for (int m = 0; m < n_in; ++m)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        // # chainer mode
        // padded[m][i+pad][j+pad] = input[m][i][j];
        // # tensorflow mode
        padded[m][i+tf_pad_h][j+tf_pad_w] = input[m][i][j];

  return padded;
}

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

  auto padded = make_pad(input, weight, stride, pad);

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
}

template <typename T>
static void im2col(Mat4D<T>& x_a, Mat3D<T>& w_a,
                   Mat3D<T>& y, Mat3D<T>& x, Mat4D<T>& w,
                   int stride, int pad)
{
  const int n_out = y.size();
  const int out_h = y[0].size();
  const int out_w = y[0][0].size();

  const int n_in = x.size();
  const int in_h = x[0].size();
  const int in_w = x[0][0].size();

  const int fil_h = w[0][0].size();
  const int fil_w = w[0][0][0].size();
  const int n_fil = fil_h * fil_w;

  w_a = zeros<T>(n_out, n_in, n_fil);

  for (int i = 0; i < n_out; ++i)
    for (int j = 0; j < n_in; ++j)
      for (int l = 0; l < fil_h; ++l)
        for (int m = 0; m < fil_w; ++m)
          w_a[i][j][fil_w*l+m] = w[i][j][l][m];

  auto padded = make_pad(x, w, stride, pad);

  const int fea_h = in_h + 2*pad - fil_h + 1;
  const int fea_w = in_w + 2*pad - fil_w + 1;

  x_a = zeros<T>(n_in, out_h, out_w, n_fil);
  for (int j = 0; j < n_in; ++j)
    for (int k = 0; k < fea_h; k+=stride)
      for (int l = 0; l < fea_w; l+=stride)
        for (int m = 0; m < fil_h; ++m)
          for (int n = 0; n < fil_w; ++n)
            x_a[j][k/stride][l/stride][fil_w*m+n] = padded[j][k+m][l+n];
}

template <typename T>
void conv_aligned(Mat3D<T>& output, Mat3D<T>& input, Mat4D<T>& weight,
                  int stride, int pad)
{
  const int n_out = output.size();
  const int n_in = input.size();
  const int fil_h = weight[0][0].size();
  const int fil_w = weight[0][0][0].size();
  const int n_fil = fil_h * fil_w;

  const int out_h = output[0].size();
  const int out_w = output[0][0].size();

  Mat4D<T> input_aligned;
  Mat3D<T> weight_aligned;
  im2col(input_aligned, weight_aligned, output, input, weight, stride, pad);

  for (int i = 0; i < n_out; ++i)
    for (int j = 0; j < n_in; ++j)
      for (int k = 0; k < out_h; ++k)
        for (int l = 0; l < out_w; ++l) {
          T acc = 0.0;
          for (int m = 0; m < n_fil; ++m)
            acc += mlt(input_aligned[j][k][l][m], weight_aligned[i][j][m]);
            // acc += input_aligned[j][k][l][m] * weight_aligned[i][j][m];
          output[i][k][l] += acc;
        }
}

template <typename T>
void conv_gemm(Mat3D<T>& output, Mat3D<T>& input, Mat4D<T>& weight,
               int stride, int pad)
{
}

#endif
