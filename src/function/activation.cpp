#ifdef _ACTV_HPP_

#include <cmath>
#include <limits>

template <typename T>
void relu(Mat3D<T>& output, Mat3D<T>& input)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_in; ++n)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        if (input[n][i][j] < 0)
          output[n][i][j] = 0;
        else
          output[n][i][j] = input[n][i][j];
}

template <typename T>
void relu(Mat1D<T>& output, Mat1D<T>& input)
{
  const int n_in = input.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_in; ++n)
    if (input[n] < 0)
      output[n] = 0;
    else
      output[n] = input[n];
}

template <typename T>
void softmax(Mat1D<T>& output, Mat1D<T>& input)
{
  const int len = input.size();

  // TODO: implement
  for (int i = 0; i < len; ++i)
    output[i] = input[i];
}

#include <iostream>
void softmax(Mat1D<float>& output, Mat1D<float>& input)
{
  const int len = input.size();

  float expsum = 0.0;
  for (int i = 0; i < len; ++i)
    expsum += exp(input[i]);

  if (std::abs(expsum-0.0) < std::numeric_limits<float>::epsilon())
    throw "softmax calculation failed";

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < len; ++i) {
    output[i] = exp(input[i]) / expsum;
    // NOTE: avoid inf / inf
    if (std::isnan(output[i]))
      output[i] = 1.0;
  }
}

void sigmoid(Mat1D<float>& output, Mat1D<float>& input)
{
  const int len = input.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < len; ++i) {
    // output[i] = (1.0/(1.0 + exp(-input[i]))) * Q_OFFSET<float>;
    output[i] = (1.0/(1.0 + exp(-input[i])));
  }
}

#endif
