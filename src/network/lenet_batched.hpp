#ifndef _LENET_HPP_
#define _LENET_HPP_

template <typename T>
class LeNet : Network<T, int>
{
private:
  const int FWID = 5;
  const int FHEI = 5;
  const int IMWID = 28;
  const int IMHEI = 28;
  const int PWID = 2;
  const int PHEI = 2;
  const int N_F1 = 16;
  const int N_F2 = 32;
  const int N_HL = 128;
  const int LABEL = 10;
  const int pm1hei = (IMHEI-FHEI+1)/PHEI;
  const int pm1wid = (IMWID-FWID+1)/PWID;
  const int pm2hei = (pm1hei-FHEI+1)/PHEI;
  const int pm2wid = (pm1wid-FWID+1)/PWID;

  // BinaryConv2D<T> conv1;
  // BinaryConv2D<T> conv2;
  Convolution2D<T> conv1;
  Convolution2D<T> conv2;
  MaxPooling<T> pool1;
  MaxPooling<T> pool2;
  // BST<T> relu1;
  // BST<T> relu2;
  // BST<T> relu3;
  Rectifier<T> relu1;
  Rectifier<T> relu2;
  Rectifier<T> relu3;
  // BinaryFull<T> full3;
  // BinaryFull<T> full4;
  FullyConnected<T> full3;
  FullyConnected<T> full4;
  SoftMax<T> output4;
  BatchNormalization<T> norm1;
  BatchNormalization<T> norm2;
  BatchNormalization<T> norm3;
  BatchNormalization<T> norm4;

  Mat3D<T> input;
  Mat3D<T> fmap1;
  Mat3D<T> bmap1;
  Mat3D<T> pmap1;
  Mat3D<T> amap1;
  Mat3D<T> fmap2;
  Mat3D<T> bmap2;
  Mat3D<T> pmap2;
  Mat3D<T> amap2;
  Mat1D<T> amap2_flat;
  Mat1D<T> hunit;
  Mat1D<T> bunit;
  Mat1D<T> aunit;
  Mat1D<T> output;
  Mat1D<T> bout;

public:
  LeNet();
  ~LeNet();

  void Load(std::string path);
  void Save(std::string path);

  void Forward(std::string data);
  void Backward(int label);
  void Update();

  int calc(std::string data, int which, int amount);
};

#include "lenet_batched.cpp"
#endif
