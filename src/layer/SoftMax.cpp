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
void SoftMax<T>::forward(Mat1D<T> &input, Mat1D<T> &output)
{
  const int len = input.size();

  double expsum = 0.0;

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < len; i++)
    expsum += exp((double)input[i]);

  if (std::abs(expsum-0.0) < std::numeric_limits<T>::epsilon())
    throw "Softmax.forward calculation failed";

  for (int i = 0; i < len; i++)
    output[i] = exp((double)input[i])/expsum;
}

template <typename T>
void SoftMax<T>::backward(Mat1D<T> &output, Mat1D<T> &input)
{
  //const int len = output.size();
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

#endif
