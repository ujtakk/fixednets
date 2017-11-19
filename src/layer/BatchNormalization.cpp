#ifdef _BATCH_HPP_

#include <iostream>
#include <cmath>

#include "fixed_point.hpp"

template <typename T>
BatchNormalization<T>::BatchNormalization(int channels)
  : shape(channels)
{
  gamma = zeros<T>(channels);
  beta  = zeros<T>(channels);
  mean  = zeros<T>(channels);
  var   = zeros<T>(channels);
  std   = zeros<T>(channels);
}

template <typename T>
BatchNormalization<T>::~BatchNormalization()
{
}

template <typename T>
void BatchNormalization<T>::load(string path)
{
  load_txt(gamma, path+"/gamma.txt");
  load_txt(beta, path+"/beta.txt");
  load_txt(eps, path+"/eps.txt");
  load_txt(mean, path+"/mean.txt");
  load_txt(var, path+"/var.txt");
  // load_bn(path+"/gamma.txt", gamma, shape);
  // load_bn(path+"/beta.txt", beta, shape);
  // load_bn(path+"/mean.txt", mean, shape);
  // load_bn(path+"/var.txt", var, shape);
  // load_eps(path+"/eps.txt", eps);

  for (int i=0; i<shape; i++)
    std[i] = sqrt((double)(var[i]*Q_OFFSET<T> + eps*Q_OFFSET<T>));
}

template <typename T>
void BatchNormalization<T>::save(string path)
{
}

template <typename T>
void BatchNormalization<T>::forward(Mat1D<T>& input, Mat1D<T>& output)
{
  const int ilen = input.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i=0; i<ilen; i++) {
    output[i] = gamma[i] * ( (input[i] - mean[i]) * Q_OFFSET<T> / std[i] )
              / Q_OFFSET<T> + beta[i];
  }
}

template <typename T>
void BatchNormalization<T>::forward(Mat3D<T>& input, Mat3D<T>& output)
{
  const int inum = input.size();
  const int ihei = input[0].size();
  const int iwid = input[0][0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n=0; n<inum; n++) {
    for (int i=0; i<ihei; i++) {
      for (int j=0; j<iwid; j++) {
        output[n][i][j] = gamma[n]
                        * ( (input[n][i][j] - mean[n]) * Q_OFFSET<T> / std[n] )
                        / Q_OFFSET<T> + beta[n];
      }
    }
  }
}

template <typename T>
void BatchNormalization<T>::backward()
{
}

// template <typename T>
// void BatchNormalization<T>::backward()
// {
// }

template <typename T>
void BatchNormalization<T>::update()
{
}

#endif
