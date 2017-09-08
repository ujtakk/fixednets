#ifdef _CONVOLUTION2D_HPP_

#include "func.hpp"
#include "load.hpp"

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
  this->pad  = pad;
}

template <typename T>
Convolution2D<T>::~Convolution2D()
{
}

template <typename T>
void Convolution2D<T>::load(string path)
{
  vector<string> filename(shape[0]);

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
}

template <typename T>
void Convolution2D<T>::save(string path)
{
}

template <typename T>
void Convolution2D<T>::forward(Mat3D<T> &input, Mat3D<T> &output)
{
  const int ihei = input[0].size();
  const int iwid = input[0][0].size();

  Mat3D<T> sum;
  Mat3D<T> pro;

  pro = zeros<T>(shape[0],
                 (ihei-shape[2]+stride+2*pad)/stride,
                 (iwid-shape[3]+stride+2*pad)/stride);
  sum = zeros<T>(shape[0],
                 (ihei-shape[2]+stride+2*pad)/stride,
                 (iwid-shape[3]+stride+2*pad)/stride);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      conv_plus_pad(input[j], iw[i][j], pro[i],
                    ihei, iwid, shape[2], shape[3],
                    stride, pad);

      for (int k = 0; k < (ihei+2*pad-shape[2]+stride)/stride; k++) {
        for (int l = 0; l < (iwid+2*pad-shape[3]+stride)/stride; l++) {
          sum[i][k][l] += pro[i][k][l];
        }
      }
    }

    for (int k = 0; k < (ihei-shape[2]+stride+2*pad)/stride; k++) {
      for (int l = 0; l < (iwid-shape[3]+stride+2*pad)/stride; l++) {
        output[i][k][l] = sum[i][k][l] + ib[i];
        sum[i][k][l] = 0;
      }
    }
  }
}

template <typename T>
void Convolution2D<T>::backward(Mat3D<T> &output, Mat3D<T> &input)
{
}

template <typename T>
void Convolution2D<T>::update()
{
}

#endif
