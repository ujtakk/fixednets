#ifndef _CIFAR_HPP_
#define _CIFAR_HPP_

#include "base.hpp"
#include "layer.hpp"

template <typename T>
class CifarNet : Network<T, int>
{
private:
  const int FWID = 5;
  const int FHEI = 5;
  const int IMWID = 32;
  const int IMHEI = 32;
  const int PWID = 2;
  const int PHEI = 2;
  const int N_F1 = 64;
  const int N_F2 = 128;
  const int N_F3 = 256;
  const int N_F4 = 512;
  const int N_F5 = 512;
  const int N_H1 = 512;
  const int LABEL = 10;

  const int pm1hei = IMHEI / PHEI;
  const int pm1wid = IMWID / PWID;
  const int pm2hei = pm1hei / PHEI;
  const int pm2wid = pm1wid / PWID;
  const int pm3hei = pm2hei / PHEI;
  const int pm3wid = pm2wid / PWID;
  const int pm4hei = pm3hei / PHEI;
  const int pm4wid = pm3wid / PWID;
  const int pm5hei = pm4hei / PHEI;
  const int pm5wid = pm4wid / PWID;

  Convolution2D<T> conv1;
  Convolution2D<T> conv2;
  Convolution2D<T> conv3;
  Convolution2D<T> conv4;
  Convolution2D<T> conv5;
  MaxPooling<T> pool1;
  MaxPooling<T> pool2;
  MaxPooling<T> pool3;
  MaxPooling<T> pool4;
  MaxPooling<T> pool5;
  Rectifier<T> relu1;
  Rectifier<T> relu2;
  Rectifier<T> relu3;
  Rectifier<T> relu4;
  Rectifier<T> relu5;
  Rectifier<T> relu6;
  FullyConnected<T> full6;
  FullyConnected<T> full7;
  //SoftMax output6;

  Mat3D<T> input;
  Mat3D<T> fmap1;
  Mat3D<T> pmap1;
  Mat3D<T> amap1;
  Mat3D<T> fmap2;
  Mat3D<T> pmap2;
  Mat3D<T> amap2;
  Mat3D<T> fmap3;
  Mat3D<T> pmap3;
  Mat3D<T> amap3;
  Mat3D<T> fmap4;
  Mat3D<T> pmap4;
  Mat3D<T> amap4;
  Mat3D<T> fmap5;
  Mat3D<T> pmap5;
  Mat3D<T> amap5;
  Mat1D<T> amap5_flat;
  Mat1D<T> hunit1;
  Mat1D<T> aunit1;
  Mat1D<T> output;

public:
  CifarNet();
  ~CifarNet();

  void Load(std::string path);
  void Save(std::string path);

  void Forward(std::string data);
  void Backward(int label);
  void Update();

  int calc(std::string data);
};

#include "cifar_net.cpp"
#endif
