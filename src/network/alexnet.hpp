#ifndef _ALEXNET_HPP_
#define _ALEXNET_HPP_

template <typename T>
class AlexNet : Network<T, vector<int>>
{
private:
  const int INSIZE = 227;
  const int PSIZE  = 3;
  const int FSIZE1 = 11;
  const int STRID1 = 4;
  const int STRID2 = 2;
  const int FSIZE2 = 5;
  const int FSIZE3 = 3;
  const int N_F1   = 96;
  const int N_F2   = 256;
  const int N_F3   = 384;
  const int N_F4   = 384;
  const int N_F5   = 256;
  const int N_H1   = 4096;
  const int N_H2   = 4096;
  const int LABEL  = 1000;

  const int fm1hei = (INSIZE-FSIZE1+STRID1)/STRID1; //55
  const int fm1wid = fm1hei;
  const int pm1hei = (fm1hei-PSIZE+STRID2)/STRID2; //27
  const int pm1wid = pm1hei;
  const int pm2hei = (pm1hei-PSIZE+STRID2)/STRID2;
  const int pm2wid = pm2hei;
  const int pm5hei = (pm2hei-PSIZE+STRID2)/STRID2;
  const int pm5wid = pm5hei;

#if defined _EAGER
  Convolution2D<T> conv1;
  Convolution2D<T> conv2;
  Convolution2D<T> conv3;
  Convolution2D<T> conv4;
  Convolution2D<T> conv5;
  MaxPooling<T> pool1;
  MaxPooling<T> pool2;
  MaxPooling<T> pool5;
#elif defined _LAZY
  /* TODO: Handle BN in LCP (It may be tough) */
  //lcpPAD<T> convpool1;
  //lcpPAD<T> convpool2;
  Convolution2D<T> conv1;
  Convolution2D<T> conv2;
  Convolution2D<T> conv3;
  Convolution2D<T> conv4;
  lcpPAD<T> convpool5;
  MaxPooling<T> pool1;
  MaxPooling<T> pool2;
#endif
  Rectifier<T> relu1;
  Rectifier<T> relu2;
  Rectifier<T> relu3;
  Rectifier<T> relu4;
  Rectifier<T> relu5;
  Rectifier<T> relu6;
  Rectifier<T> relu7;
  FullyConnected<T> full6;
  FullyConnected<T> full7;
  FullyConnected<T> full8;
  BatchNormalization<T> bn1;
  BatchNormalization<T> bn2;

  Mat3D<T> input;
  Mat3D<T> fmap1;
  Mat3D<T> bmap1;
  Mat3D<T> pmap1;
  Mat3D<T> amap1;
  Mat3D<T> fmap2;
  Mat3D<T> bmap2;
  Mat3D<T> pmap2;
  Mat3D<T> amap2;
  Mat3D<T> fmap3;
  Mat3D<T> amap3;
  Mat3D<T> fmap4;
  Mat3D<T> amap4;
  Mat3D<T> fmap5;
  Mat3D<T> pmap5;
  Mat3D<T> amap5;
  Mat1D<T> amap5_flat;
  Mat1D<T> hunit1;
  Mat1D<T> aunit1;
  Mat1D<T> hunit2;
  Mat1D<T> aunit2;
  Mat1D<T> output;

public:
  AlexNet();
  ~AlexNet();

  void Load(string path);
  vector<int> calc(string data);
};

#include "alexnet.cpp"
#endif
