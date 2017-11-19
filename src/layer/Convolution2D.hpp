#ifndef _CONVOLUTION2D_HPP_
#define _CONVOLUTION2D_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

using std::string;

template <typename T>
class Convolution2D : ParamLayer<Mat3D<T>>
{
private:

public:
  Convolution2D(int n_out, int n_in, int fil_h, int fil_w, int stride=1, int pad=0);
  ~Convolution2D();

  Mat4D<T> iw;
  Mat4D<T> gw;
  Mat1D<T> ib;
  Mat1D<T> gb;

  const int shape[4];
  int stride;
  int pad;

  void load(string path);
  void save(string path);

  void forward(Mat3D<T>& input, Mat3D<T>& output);
  void backward(Mat3D<T>& output, Mat3D<T>& input);

  void update();
};

#include "Convolution2D.cpp"
#endif
