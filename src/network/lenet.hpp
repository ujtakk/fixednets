#ifndef _LENET_HPP_
#define _LENET_HPP_

#include "layer.hpp"

template <typename T>
class LeNet : Network<T, int>
{
private:
  const int FWID  = 5;
  const int FHEI  = 5;
  const int IMWID = 28;
  const int IMHEI = 28;
  const int PWID  = 2;
  const int PHEI  = 2;
  const int N_F1  = 16;
  const int N_F2  = 32;
  const int N_HL  = 256;
  const int LABEL = 10;
  const int pm1hei = (IMHEI-FHEI+1)/PHEI;
  const int pm1wid = (IMWID-FWID+1)/PWID;
  const int pm2hei = (pm1hei-FHEI+1)/PHEI;
  const int pm2wid = (pm1wid-FWID+1)/PWID;
  // const int pm1hei = IMHEI/PHEI;
  // const int pm1wid = IMWID/PWID;
  // const int pm2hei = pm1hei/PHEI;
  // const int pm2wid = pm1wid/PWID;

  Convolution2D<T>  conv1;
  Convolution2D<T>  conv2;
  MaxPooling<T>     pool1;
  MaxPooling<T>     pool2;
  Rectifier<T>      relu1;
  Rectifier<T>      relu2;
  Rectifier<T>      relu3;
  FullyConnected<T> full3;
  FullyConnected<T> full4;
  SoftMax<T>        prob4;

  Mat3D<T> input;
  Mat3D<T> fmap1;
  Mat3D<T> pmap1;
  Mat3D<T> amap1;
  Mat3D<T> fmap2;
  Mat3D<T> pmap2;
  Mat3D<T> amap2;
  Mat1D<T> pmap2_flat;
  Mat1D<T> fvec3;
  Mat1D<T> avec3;
  Mat1D<T> fvec4;
  Mat1D<T> output;

public:
  LeNet();
  ~LeNet();

  void Load(std::string path);
  void Save(std::string path);

  void Forward(std::string data);
  void Backward(int label);
  void Update();

  int calc(std::string data);
};

#include "lenet.cpp"
#endif
