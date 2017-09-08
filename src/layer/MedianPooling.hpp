#ifndef _MEDIANPOOLING_HPP_
#define _MEDIANPOOLING_HPP_

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class MedianPooling : Layer<Mat3D<T>>
{
private:

public:
  MedianPooling(const int phei, const int pwid);
  ~MedianPooling();

  const int shape[2];

  void forward(Mat3D<T> &input, Mat3D<T> &output);
  void backward(Mat3D<T> &output, Mat3D<T> &input);
};

#include "MedianPooling.cpp"
#endif
