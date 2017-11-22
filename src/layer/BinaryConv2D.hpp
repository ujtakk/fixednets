#ifndef _BINARYCONV2D_HPP_
#define _BINARYCONV2D_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class BinaryConv2D : ParamLayer<Mat3D<T>>
{
private:

public:
  BinaryConv2D(int out_channels, int in_channels,
                 const int f_height, const int f_width,
                 int stride=1, int pad=0);
  ~BinaryConv2D();

  Mat4D<T> iw;
  Mat4D<T> gw;
  Mat1D<T> ib;
  Mat1D<T> gb;

  const int shape[4];
  int stride;
  int pad;

  void load(std::string path);
  void save(std::string path);

  void forward(Mat3D<T>& input, Mat3D<T>& output);
  void backward(Mat3D<T>& output, Mat3D<T>& input);

  void update();
};

#include "BinaryConv2D.hpp"
#endif
