#ifdef _CONVOLUTION2D_HPP_

#include "load.hpp"
#include "function.hpp"

using std::to_string;

template <typename T>
Convolution2D<T>::Convolution2D(int n_out, int n_in, const int fil_h, const int fil_w, int stride, int pad)
  : shape{n_out, n_in, fil_h, fil_w}
{
  iw = zeros<T>(n_out, n_in, fil_h, fil_w);
  gw = zeros<T>(n_out, n_in, fil_h, fil_w);
  ib = zeros<T>(n_out);
  gb = zeros<T>(n_out);
  this->stride = stride;
  this->pad    = pad;
}

template <typename T>
Convolution2D<T>::~Convolution2D()
{
}

template <typename T>
void Convolution2D<T>::load(string path)
{
#if 1
  load_txt(iw, path+"/W.txt");
  load_txt(ib, path+"/b.txt");
#else
  std::vector<string> filename(shape[0]);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      filename[i] = path+"/data"+to_string(i)+"_"+to_string(j)+".txt";
      load_w(filename[i], iw[i][j], shape[2], shape[3]);
    }

    filename[i] = path+"/data"+to_string(i)+".txt";
    load_b(filename[i], ib[i]);
  }
#endif
}

template <typename T>
void Convolution2D<T>::save(string path)
{
  save_txt(iw, path+"/W.txt");
  save_txt(ib, path+"/b.txt");
}

template <typename T>
void Convolution2D<T>::forward(Mat3D<T>& input, Mat3D<T>& output)
{
  const int n_out = output.size();
  const int out_h = output[0].size();
  const int out_w = output[0][0].size();

  Mat3D<T> conved = zeros<T>(n_out, out_h, out_w);

  conv_plus_pad(input, iw, conved, stride, pad);
  bias(conved, ib, output);
}

template <typename T>
void Convolution2D<T>::backward(Mat3D<T>& output, Mat3D<T>& input)
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
}

#endif
