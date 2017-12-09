#ifndef _FIREMODULE_HPP_
#define _FIREMODULE_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class FireModule : ParamLayer<Mat3D<T>>
{
private:

public:
  FireModule(int s1x1, int e1x1, int e3x3, int n_in,
             bool quantized=false);
  ~FireModule();

  ConvModule<T> squeeze1x1;
  ConvModule<T> expand1x1;
  ConvModule<T> expand3x3;

  bool quantized;

  int s1x1, e1x1, e3x3, n_in;

  void load(std::string path);
  void save(std::string path);

  void forward(Mat3D<T>& output, Mat3D<T>& input);
  void backward(Mat3D<T>& input, Mat3D<T>& output);

  void update();
};

#include "FireModule.cpp"
#endif
