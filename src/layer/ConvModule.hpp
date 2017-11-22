#ifndef _CONVMODULE_HPP_
#define _CONVMODULE_HPP_

#include "base.hpp"
#include "matrix.hpp"
#include "layer.hpp"

template <typename T>
class ConvModule : ParamLayer<Mat3D<T>>
{
private:
public:
  ConvModule(int n_out, int n_in, int fil_h, int fil_w,
             int stride=1, int pad=0);
  ~ConvModule();

  Convolution2D<T> conv;
  Rectifier<T>     relu;

  void load(std::string path);
  void save(std::string path);

  void forward(Mat3D<T>& output, Mat3D<T>& input);
  void backward(Mat3D<T>& input, Mat3D<T>& output);

  void update();
};

#include "ConvModule.cpp"
#endif
