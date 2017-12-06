#ifdef _BATCH_HPP_

#include <iostream>
#include <cmath>

#include "types.hpp"

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
void BatchNormalization<T>::load(std::string path)
{
  load_txt(gamma, path+"/gamma.txt");
  load_txt(beta, path+"/beta.txt");
  load_txt(eps, path+"/eps.txt");
  load_txt(mean, path+"/mean.txt");
  load_txt(var, path+"/var.txt");

  for (int i=0; i<shape; i++)
    // std[i] = sqrt((double)(var[i]*Q_OFFSET<T> + eps*Q_OFFSET<T>));
    std[i] = sqrt(static_cast<double>(to_fixed(var[i] + eps)));
}

template <typename T>
void BatchNormalization<T>::save(std::string path)
{
}

template <typename T>
void BatchNormalization<T>::forward(Mat1D<T>& output, Mat1D<T>& input)
{
  norm_batch(output, input, gamma, beta, eps, mean, var);
}

template <typename T>
void BatchNormalization<T>::forward(Mat3D<T>& output, Mat3D<T>& input)
{
  norm_batch(output, input, gamma, beta, eps, mean, var);
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
