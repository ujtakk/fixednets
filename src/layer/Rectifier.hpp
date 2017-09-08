#ifndef _RECTIFIER_HPP_
#define _RECTIFIER_HPP_

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class Rectifier : Layer<Mat1D<T>>
{
private:

public:
  Rectifier();
  ~Rectifier();

  void forward(Mat3D<T> &input, Mat3D<T> &output);
  void forward(Mat1D<T> &input, Mat1D<T> &output);
  void backward(Mat3D<T> &output, Mat3D<T> &input);
  void backward(Mat1D<T> &output, Mat1D<T> &input);
};

#include "Rectifier.cpp"
#endif
