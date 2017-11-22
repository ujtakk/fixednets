#ifndef _VGG_HPP_
#define _VGG_HPP_

#include <vector>

// #define _LAZY

template <typename T>
class VGGNet : Network<T, std::vector<int>>
{
private:
  const int INSIZE = 224;
  const int FSIZE  = 3;
  const int FSTRID = 1;
  const int FPAD   = 1;
  const int PSIZE  = 2;
  const int PSTRID = 2;
  const int N_F1   = 64;
  const int N_F2   = 128;
  const int N_F3   = 256;
  const int N_F4   = 512;
  const int N_F5   = 512;
  const int N_H1   = 4096;
  const int N_H2   = 4096;
  const int LABEL  = 1000;

  const int fm1hei = INSIZE; //(INSIZE-FSIZE+FSTRID)/FSTRID;
  const int fm1wid = fm1hei;
  const int pm1hei = fm1hei/PSTRID; //(fm1hei-PSIZE+PSTRID)/PSTRID;
  const int pm1wid = pm1hei;
  const int pm2hei = pm1hei/PSTRID; //(pm1hei-PSIZE+STRID2)/STRID2;
  const int pm2wid = pm2hei;
  const int pm3hei = pm2hei/PSTRID; //(pm1hei-PSIZE+STRID2)/STRID2;
  const int pm3wid = pm3hei;
  const int pm4hei = pm3hei/PSTRID; //(pm1hei-PSIZE+STRID2)/STRID2;
  const int pm4wid = pm4hei;
  const int pm5hei = pm4hei/PSTRID; //(pm2hei-PSIZE+STRID2)/STRID2;
  const int pm5wid = pm5hei;

#ifdef _LAZY
  Convolution2D<T> conv1_1;
  lcpPAD convpool1;

  Convolution2D<T> conv2_1;
  lcpPAD convpool2;

  Convolution2D<T> conv3_1;
  Convolution2D<T> conv3_2;
  lcpPAD convpool3;

  Convolution2D<T> conv4_1;
  Convolution2D<T> conv4_2;
  lcpPAD convpool4;

  Convolution2D<T> conv5_1;
  Convolution2D<T> conv5_2;
  lcpPAD convpool5;

  FullyConnected<T> full6;
  FullyConnected<T> full7;
  FullyConnected<T> full8;
#else
  Convolution2D<T> conv1_1;
  Convolution2D<T> conv1_2;
  MaxPooling<T> pool1;

  Convolution2D<T> conv2_1;
  Convolution2D<T> conv2_2;
  MaxPooling<T> pool2;

  Convolution2D<T> conv3_1;
  Convolution2D<T> conv3_2;
  Convolution2D<T> conv3_3;
  MaxPooling<T> pool3;

  Convolution2D<T> conv4_1;
  Convolution2D<T> conv4_2;
  Convolution2D<T> conv4_3;
  MaxPooling<T> pool4;

  Convolution2D<T> conv5_1;
  Convolution2D<T> conv5_2;
  Convolution2D<T> conv5_3;
  MaxPooling<T> pool5;

  FullyConnected<T> full6;
  FullyConnected<T> full7;
  FullyConnected<T> full8;
#endif

  Rectifier<T> relu1_1;
  Rectifier<T> relu1_2;
  Rectifier<T> relu2_1;
  Rectifier<T> relu2_2;
  Rectifier<T> relu3_1;
  Rectifier<T> relu3_2;
  Rectifier<T> relu3_3;
  Rectifier<T> relu4_1;
  Rectifier<T> relu4_2;
  Rectifier<T> relu4_3;
  Rectifier<T> relu5_1;
  Rectifier<T> relu5_2;
  Rectifier<T> relu5_3;
  Rectifier<T> relu6;
  Rectifier<T> relu7;
  Rectifier<T> relu8;

  Mat3D<T> input;
  Mat3D<T> fmap1_1;
  Mat3D<T> amap1_1;
  Mat3D<T> fmap1_2;
  Mat3D<T> amap1_2;
  Mat3D<T> pmap1;

  Mat3D<T> fmap2_1;
  Mat3D<T> amap2_1;
  Mat3D<T> fmap2_2;
  Mat3D<T> amap2_2;
  Mat3D<T> pmap2;

  Mat3D<T> fmap3_1;
  Mat3D<T> amap3_1;
  Mat3D<T> fmap3_2;
  Mat3D<T> amap3_2;
  Mat3D<T> fmap3_3;
  Mat3D<T> amap3_3;
  Mat3D<T> pmap3;

  Mat3D<T> fmap4_1;
  Mat3D<T> amap4_1;
  Mat3D<T> fmap4_2;
  Mat3D<T> amap4_2;
  Mat3D<T> fmap4_3;
  Mat3D<T> amap4_3;
  Mat3D<T> pmap4;

  Mat3D<T> fmap5_1;
  Mat3D<T> amap5_1;
  Mat3D<T> fmap5_2;
  Mat3D<T> amap5_2;
  Mat3D<T> fmap5_3;
  Mat3D<T> amap5_3;
  Mat3D<T> pmap5;

  Mat1D<T> amap5_flat;
  Mat1D<T> hunit1;
  Mat1D<T> aunit1;
  Mat1D<T> hunit2;
  Mat1D<T> aunit2;
  Mat1D<T> output;

public:
  VGGNet();
  ~VGGNet();

  void Load(std::string path);
  std::vector<int> calc(std::string data);
};

#include "vgg.cpp"
#undef _LAZY
#endif
