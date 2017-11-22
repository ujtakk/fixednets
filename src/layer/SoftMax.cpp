#ifdef _SOFTMAX_HPP_

template <typename T>
SoftMax<T>::SoftMax()
{
}

template <typename T>
SoftMax<T>::~SoftMax()
{
}

template <typename T>
void SoftMax<T>::forward(Mat1D<T>& input, Mat1D<T>& output)
{
  softmax(input, output);
}

template <typename T>
void SoftMax<T>::backward(Mat1D<T>& output, Mat1D<T>& input)
{
  //const int len = output.size();
}

template <typename T>
Mat1D<T> SoftMax<T>::forward(Mat1D<T>& input)
{
  auto output = softmax(input);

  return output;
}

template <typename T>
Mat1D<T> SoftMax<T>::backward(Mat1D<T>& output)
{
  const int len = output.size();
  auto input = zeros<T>(len);

  return input;
}

template <typename T>
double SoftMax<T>::negative_log_likelihood(Mat1D<T> y)
{
  //mean of log probabilities over labels of each example.
  double a = 0.0;

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < (int)y.size(); i++)
    a += log(p_y_given_x[i][y[i]]);

  return -a / (double)y.size();
}

template <typename T>
double SoftMax<T>::errors(Mat1D<T> y)
{
  //mean distance between predicted labels and supervised labels.
  double a = 0.0;

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < (int)y.size(); i++)
    if (y_pred[i] != y[i]) a += 1.0;

  return a / (double)y.size();
}

template <typename T>
void SoftMax<T>::prob(Mat1D<T>& input, Mat1D<T>& output)
{
  softmax(input, output);
}

#include <cstdio>
template <typename T>
void SoftMax<T>::loss(int label, Mat1D<T>& output, Mat1D<T>& input)
{
  const int olen = output.size();
  // const int ilen = input.size();
  Mat1D<T> truth = zeros<T>(olen);
  truth[label] = 1.0;

  for (int i = 0; i < olen; ++i) {
    input[i] = output[i] - truth[i];
    // printf("%7.3f", input[i]);
  }
  // printf("\n");
}

template <typename T>
Mat1D<T> SoftMax<T>::loss(Mat1D<T>& output, int label)
{
  const int len = output.size();
  auto input = zeros<T>(len);
  auto truth = zeros<T>(len);
  truth[label] = 1.0;

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < len; ++i) {
    input[i] = output[i] - truth[i];
    // printf("%7.3f", input[i]);
  }
  // printf("\n");

  return input;
}

#endif
