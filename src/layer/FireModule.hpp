#ifndef _FIREMODULE_HPP_
#define _FIREMODULE_HPP_

#include <string>

#include "base.hpp"
#include "matrix.hpp"

using std::string;

template <typename T>
class FireModule : ParamLayer<Mat3D<T>>
{
private:

public:
  FireModule(int s1x1, int e1x1, int e3x3, int n_in);
  ~FireModule();

  ConvModule<T> squeeze1x1;
  ConvModule<T> expand1x1;
  ConvModule<T> expand3x3;

  int s1x1, e1x1, e3x3, n_in;

  void load(string path);
  void save(string path);

  void forward(Mat3D<T>& input, Mat3D<T>& output);
  void backward(Mat3D<T>& output, Mat3D<T>& input);

  void update();
};

#include "FireModule.cpp"
#endif
