#ifndef _SIGMOID_HPP_
#define _SIGMOID_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

using std::string;

template <typename T>
class Sigmoid : Layer<Mat1D<T>>
{
private:

public:
  Sigmoid();
  ~Sigmoid();

  Mat2D<double> p_y_given_x;

  Mat1D<T> y_pred;

  void forward(Mat1D<T> &input, Mat1D<T> &output);
  void backward(Mat1D<T> &output, Mat1D<T> &input);

  double negative_log_likelihood(Mat1D<T> y);
  void errors(Mat1D<T> input, Mat1D<T> output, int label);
};

#include "Sigmoid.cpp"
#endif
