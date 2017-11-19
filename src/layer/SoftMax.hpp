#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

using std::string;

template <typename T>
class SoftMax : Layer<Mat1D<T>>
{
private:

public:
  SoftMax();
  ~SoftMax();

  Mat2D<double> p_y_given_x;

  Mat1D<T> y_pred;

  void forward(Mat1D<T>& input, Mat1D<T>& output);
  void backward(Mat1D<T>& output, Mat1D<T>& input);

  double negative_log_likelihood(Mat1D<T> y);
  double errors(Mat1D<T> y);
};

#include "SoftMax.cpp"
#endif
