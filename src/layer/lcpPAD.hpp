#ifndef _LCPPAD_HPP_
#define _LCPPAD_HPP_

#include "matrix.hpp"

template <typename T>
class lcpPAD : ParamLayer<Mat3D<T>>
{
private:

public:
  lcpPAD(int out_channels, int in_channels, const int f_height, const int f_width, const int phei, const int pwid, int cstride, int cpad, int pstride);
  ~lcpPAD();

  Mat4D<T> iw;
  Mat4D<T> gw;
  Mat1D<T> ib;
  Mat1D<T> gb;

  const int cshape[4];
  const int pshape[2];

  int cstride;
  int cpad;
  int pstride;

  void load(string path);

  void forward(Mat3D<T>& input, Mat3D<T>& output, int which, int amount);
};

#include "lcpPAD.cpp"
#endif
