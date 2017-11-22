#ifndef _LAZYCONVPOOL_HPP_
#define _LAZYCONVPOOL_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class LazyConvPool : ParamLayer<Mat3D<T>>
{
private:
public:
  LazyConvPool(int out_channels, int in_channels, const int f_height, const int f_width, const int phei, const int pwid);
  ~LazyConvPool();

  Mat4D<T> iw;
  Mat4D<T> gw;
  Mat1D<T> ib;
  Mat1D<T> gb;

  const int cshape[4];
  const int pshape[2];

  void load(std::string path);

  void forward(Mat3D<T>& input, Mat3D<T>& output, int which, int amount);
};

#include "LazyConvPool.cpp"
#endif
