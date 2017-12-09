#ifndef _FULLYCONNECTED_HPP_
#define _FULLYCONNECTED_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class FullyConnected : ParamLayer<Mat1D<T>>
{
private:

public:
  FullyConnected(int n_out, int n_in, bool quantized=false);
  ~FullyConnected();

  Mat2D<T> iw;
  Mat2D<T> gw;
  Mat1D<T> ib;
  Mat1D<T> gb;

  const int shape[2];

  bool quantized;

  void load(std::string path);
  void save(std::string path);

  void forward(Mat1D<T>& output, Mat1D<T>& input);
  void backward(Mat1D<T>& input, Mat1D<T>& output);

  void update();
};

#include "FullyConnected.cpp"
#endif
