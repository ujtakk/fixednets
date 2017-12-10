#ifdef _CONVOLUTION2D_HPP_

#include "function.hpp"

template <typename T>
Convolution2D<T>::Convolution2D(int n_out, int n_in, int fil_h, int fil_w,
                                int stride, int pad, bool quantized)
  : shape{n_out, n_in, fil_h, fil_w}
  , stride(stride), pad(pad)
  , quantized(quantized)
{
  iw = zeros<T>(n_out, n_in, fil_h, fil_w);
  gw = zeros<T>(n_out, n_in, fil_h, fil_w);
  ib = zeros<T>(n_out);
  gb = zeros<T>(n_out);
}

template <typename T>
Convolution2D<T>::~Convolution2D()
{
}

template <typename T>
void Convolution2D<T>::load(std::string path)
{
  if (quantized) {
    load_quantized(iw, path, "W.txt");
    load_quantized(ib, path, "b.txt");
    // save_txt("now_conv1_kernels.txt", iw);
    // save_txt("now_conv1_biases.txt", ib);
    // exit(0);
  }
  else {
    load_txt(iw, path+"/W.txt");
    load_txt(ib, path+"/b.txt");
  }
}

template <typename T>
void Convolution2D<T>::save(std::string path)
{
  save_txt(path+"/W.txt", iw);
  save_txt(path+"/b.txt", ib);
}

template <typename T>
void Convolution2D<T>::forward(Mat3D<T>& output, Mat3D<T>& input)
{
  // const int n_out = output.size();
  // const int out_h = output[0].size();
  // const int out_w = output[0][0].size();
  const int n_out =  shape[0];
  const int out_h = (input[0].size()    + 2*pad - shape[2] + 1) / stride;
  const int out_w = (input[0][0].size() + 2*pad - shape[3] + 1) / stride;

  auto conved = zeros<T>(n_out, out_h, out_w);
  output = zeros<T>(n_out, out_h, out_w);

  conv_plus_pad(conved, input, iw, stride, pad);
  // conv_aligned(conved, input, iw, stride, pad);
  bias(output, conved, ib);
}

template <typename T>
void Convolution2D<T>::backward(Mat3D<T>& input, Mat3D<T>& output)
{
  const int ohei = output[0].size();
  const int owid = output[0][0].size();

  Mat4D<T> pro;
  Mat2D<T> sum;

  pro = zeros<T>(shape[0], shape[1], ohei+shape[2]-1, owid+shape[3]-1);
  sum = zeros<T>(ohei+shape[2]-1, owid+shape[3]-1);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[1]; i++) {
    for (int j = 0; j < shape[0]; j++) {
      //conv_valid(input[i], output[j], gw[j][i], ohei+shape[2]-1, owid+shape[3]-1, ohei, owid);
      //conv_full(output[j], iw[j][i], pro[j][i], ohei, owid, shape[2], shape[3]);

      for (int k = 0; k < ohei+shape[2]-1; k++) {
        for (int l = 0; l < owid+shape[3]-1; l++) {
          sum[k][l] += pro[j][i][k][l];
        }
      }
    }

    for (int k = 0; k < ohei+shape[2]-1; k++) {
      for (int l = 0; l < owid+shape[3]-1; l++) {
        input[i][k][l] = sum[k][l];
        sum[k][l] = 0;
      }
    }
  }

  for (int i = 0; i < shape[0]; i++) {
    gb[i] = 0;

    for (int k = 0; k < ohei; k++) {
      for (int l = 0; l < owid; l++) {
        gb[i] += output[i][k][l];
      }
    }
  }
}

template <typename T>
void Convolution2D<T>::update()
{
#if 0
  float alpha = 0.001;

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; ++i)
    for (int j = 0; j < shape[1]; ++j)
      iw[i][j] -= alpha * gw[i][j];

  for (int i = 0; i < shape[0]; i++) {
    ib[i] -= alpha * gb[i];
  }
#endif
}

#endif
