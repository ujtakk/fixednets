#ifndef _CONVMODULE_HPP_
#define _CONVMODULE_HPP_

template <typename T>
class ConvModule : ParamLayer<Mat3D<T>>
{
private:
public:
  ConvModule();
  ~ConvModule();

  Convolution2D<T> conv;
  Rectifier<T>     relu;

  void load(string path);
  void save(string path);

  void forward(Mat3D<T>& input, Mat3D<T>& output);
  void backward(Mat3D<T>& output, Mat3D<T>& input);

  void update();
};

#include "ConvModule.cpp"
#endif
