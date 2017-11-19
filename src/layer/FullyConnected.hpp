#ifndef _FULLYCONNECTED_HPP_
#define _FULLYCONNECTED_HPP_

#include <string>
#include "base.hpp"

using std::string;

template <typename T>
class FullyConnected : ParamLayer<Mat1D<T>>
{
private:

public:
  FullyConnected(int n_out, int n_in);
  ~FullyConnected();

  Mat2D<T> iw;
  Mat2D<T> gw;
  Mat1D<T> ib;
  Mat1D<T> gb;

  const int shape[2];

  void load(string path);
  void save(string path);

  void forward(Mat1D<T>& input, Mat1D<T>& output);
  void backward(Mat1D<T>& output, Mat1D<T>& input);

  void update();
};

#include "FullyConnected.cpp"
#endif
