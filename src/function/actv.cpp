#ifdef _ACTV_HPP_

#include <limits>

template <typename T>
void relu(Mat3D<T>& input, Mat3D<T>& output)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  for (int n = 0; n < n_in; ++n)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        if (input[n][i][j] < 0)
          output[n][i][j] = 0;
        else
          output[n][i][j] = input[n][i][j];
}

template <typename T>
void relu(Mat1D<T>& input, Mat1D<T>& output)
{
  const int n_in = input.size();

  for (int n = 0; n < n_in; ++n)
    if (input[n] < 0)
      output[n] = 0;
    else
      output[n] = input[n];
}

void softmax(Mat1D<float>& input, Mat1D<float>& output)
{
  const int len = input.size();

  float expsum = 0.0;
  for (int i = 0; i < len; ++i)
    expsum += exp(input[i]);

  if (std::abs(expsum-0.0) < std::numeric_limits<float>::epsilon())
    throw "softmax calculation failed";

  // #ifdef _OPENMP
  // #pragma omp parallel for
  // #endif
  for (int i = 0; i < len; ++i)
    output[i] = exp(input[i]) / expsum;
}

template <typename T>
Mat3D<T> relu(Mat3D<T>& input)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();
  auto output = zeros<T>(n_in, in_h, in_w);

  for (int n = 0; n < n_in; ++n)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        if (input[n][i][j] < 0)
          output[n][i][j] = 0;
        else
          output[n][i][j] = input[n][i][j];

  return output;
}

template <typename T>
Mat1D<T> relu(Mat1D<T>& input)
{
  const int n_in = input.size();
  auto output = zeros<T>(n_in);

  for (int n = 0; n < n_in; ++n)
    if (input[n] < 0)
      output[n] = 0;
    else
      output[n] = input[n];

  return output;
}

Mat1D<float> softmax(Mat1D<float>& input)
{
  const int len = input.size();
  auto output = zeros<float>(len);

  float expsum = 0.0;
  for (int i = 0; i < len; ++i)
    expsum += exp(input[i]);

  if (std::abs(expsum-0.0) < std::numeric_limits<float>::epsilon())
    throw "softmax calculation failed";

  // #ifdef _OPENMP
  // #pragma omp parallel for
  // #endif
  for (int i = 0; i < len; ++i)
    output[i] = exp(input[i]) / expsum;

  return output;
}

#endif
