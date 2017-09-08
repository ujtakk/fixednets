#ifndef _DEEPER_CIFAR_HPP_
#define _DEEPER_CIFAR_HPP_

template <typename T>
class VGG_CIFAR : Network<T, int>
{
private:
  const int FWID = 3;
  const int FHEI = 3;
  const int IMWID = 32;
  const int IMHEI = 32;
  const int PWID = 2;
  const int PHEI = 2;
  const int N_F1 = 64;
  const int N_F2 = 128;
  const int N_F3 = 256;
  const int N_F4 = 512;
  const int N_H1 = 512;
  const int LABEL = 10;

  const int pm1hei = IMHEI / PHEI;
  const int pm1wid = IMWID / PWID;
  const int pm2hei = pm1hei / PHEI;
  const int pm2wid = pm1wid / PWID;
  const int pm3hei = pm2hei / PHEI;
  const int pm3wid = pm2wid / PWID;

  Convolution2D<T> conv1;
  BatchNormalization<T> bn1;
  Rectifier<T> relu1;

  Convolution2D<T> conv2;
  BatchNormalization<T> bn1;
  Rectifier<T> relu2;
  MaxPooling<T> pool1;
  //lcpPAD<T> convpool1;

  Convolution2D<T> conv3;
  BatchNormalization<T> bn1;
  Rectifier<T> relu3;

  Convolution2D<T> conv4;
  BatchNormalization<T> bn1;
  Rectifier<T> relu4;
  MaxPooling<T> pool2;
  //lcpPAD<T> convpool2;

  Convolution2D<T> conv5;
  BatchNormalization<T> bn1;
  Rectifier<T> relu5;

  Convolution2D<T> conv6;
  BatchNormalization<T> bn1;
  Rectifier<T> relu6;

  Convolution2D<T> conv7;
  BatchNormalization<T> bn1;
  Rectifier<T> relu7;
  MaxPooling<T> pool3;
  //lcpPAD<T> convpool3;

  Convolution2D<T> conv8;
  BatchNormalization<T> bn1;
  Rectifier<T> relu5;

  Convolution2D<T> conv6;
  BatchNormalization<T> bn1;
  Rectifier<T> relu6;

  Convolution2D<T> conv7;
  BatchNormalization<T> bn1;
  Rectifier<T> relu7;
  MaxPooling<T> pool3;
  //lcpPAD<T> convpool3;

  FullyConnected<T> full9;
  BatchNormalization<T> bn1;
  Rectifier<T> relu9;

  FullyConnected<T> full10;
  //SoftMax output11;

  Mat3D<T> input;
  Mat3D<T> fmap1;
  Mat3D<T> amap1;
  Mat3D<T> fmap2;
  Mat3D<T> pmap1;
  Mat3D<T> amap2;
  Mat3D<T> fmap3;
  Mat3D<T> amap3;
  Mat3D<T> fmap4;
  Mat3D<T> pmap2;
  Mat3D<T> amap4;
  Mat3D<T> fmap5;
  Mat3D<T> amap5;
  Mat3D<T> fmap6;
  Mat3D<T> amap6;
  Mat3D<T> fmap7;
  Mat3D<T> amap7;
  Mat3D<T> fmap8;
  Mat3D<T> pmap3;
  Mat3D<T> amap8;
  Mat1D<T> amap8_flat;
  Mat1D<T> hunit1;
  Mat1D<T> aunit1;
  Mat1D<T> hunit2;
  Mat1D<T> aunit2;
  Mat1D<T> output;

public:
  VGG_CIFAR();
  ~VGG_CIFAR();

  void Load(char *path);
  int calc(char *data, int which, int amount);

};

#include "cifar_batch.cpp"
#endif
