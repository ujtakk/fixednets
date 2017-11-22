#ifndef _BINARYFULL_HPP_
#define _BINARYFULL_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class BinaryFull : ParamLayer<Mat1D<T>>
{
private:

public:
  BinaryFull(int out_channels, int in_channels);
  ~BinaryFull();

  Mat2D<T> iw;
  Mat2D<T> gw;
  Mat1D<T> ib;
  Mat1D<T> gb;

  const int shape[2];

  void load(std::string path);
  void save(std::string path);

  void forward(Mat1D<T>& input, Mat1D<T>& output);
  void backward(Mat1D<T>& output, Mat1D<T>& input);

  void update();
};

#include "BinaryFull.cpp"
#endif
