#ifndef _MAXPOOLING_HPP_
#define _MAXPOOLING_HPP_

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class MaxPooling : Layer<Mat3D<T>>
{
private:

public:
  MaxPooling(const int fil_h, const int fil_w, int stride=2);
  ~MaxPooling();

  const int shape[2];
  int stride;

  void forward(Mat3D<T>& input, Mat3D<T>& output);
  void backward(Mat3D<T>& output, Mat3D<T>& input);

  Mat3D<T> forward(Mat3D<T>& input);
  Mat3D<T> backward(Mat3D<T>& output);
};

#include "MaxPooling.cpp"
#endif
