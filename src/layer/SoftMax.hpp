#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class SoftMax : Layer<Mat1D<T>>
{
private:

public:
  SoftMax();
  ~SoftMax();

  Mat2D<double> p_y_given_x;

  Mat1D<T> y_pred;

  void forward(Mat1D<T>& output, Mat1D<T>& input);
  void backward(Mat1D<T>& input, Mat1D<T>& output);

  double negative_log_likelihood(Mat1D<T> y);
  double errors(Mat1D<T> y);

  void prob(Mat1D<T>& output, Mat1D<T>& input);
  void loss(Mat1D<T>& input, Mat1D<T>& output, int label);
};

#include "SoftMax.cpp"
#endif
