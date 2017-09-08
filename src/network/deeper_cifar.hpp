#ifndef _DEEPER_CIFAR_HPP_
#define _DEEPER_CIFAR_HPP_

template <typename T>
class Deeper : Network<T, int>
{
private:
  const int FWID = 3; /*filter width*/
  const int FHEI = 3; /*filter height*/
  const int IMWID = 32; /*image width*/
  const int IMHEI = 32; /*image height*/
  const int PWID = 2; /*pooling rate*/
  const int PHEI = 2;
  const int N_F1 = 64; /*number of feature maps 1*/
  const int N_F2 = 128; /*number of feature maps 2*/
  const int N_F3 = 256; /*number of feature maps 2*/
  const int N_H1 = 1024; /*number of hidden layer unit*/
  const int N_H2 = 1024; /*number of hidden layer unit*/
  const int LABEL = 10; /*number of class*/

  const int pm1hei = IMHEI / PHEI;
  const int pm1wid = IMWID / PWID;
  const int pm2hei = pm1hei / PHEI;
  const int pm2wid = pm1wid / PWID;
  const int pm3hei = pm2hei / PHEI;
  const int pm3wid = pm2wid / PWID;

  char filename[128];
  char path[128];

  Convolution2D<T> conv1; /* 1st conv & pool layer */
  Rectifier<T> relu1;
  Convolution2D<T> conv2; /* 1st conv & pool layer */
  MaxPooling<T> pool1;
  //lcpPAD<T> convpool1; /* 1st conv & pool layer */
  Rectifier<T> relu2;
  Convolution2D<T> conv3; /* 2nd conv & pool layer */
  Rectifier<T> relu3;
  Convolution2D<T> conv4; /* 1st conv & pool layer */
  MaxPooling<T> pool2;
  //lcpPAD<T> convpool2; /* 2nd conv & pool layer */
  Rectifier<T> relu4;
  Convolution2D<T> conv5; /* 2nd conv & pool layer */
  Rectifier<T> relu5;
  Convolution2D<T> conv6; /* 2nd conv & pool layer */
  Rectifier<T> relu6;
  Convolution2D<T> conv7; /* 2nd conv & pool layer */
  Rectifier<T> relu7;
  Convolution2D<T> conv8; /* 1st conv & pool layer */
  MaxPooling<T> pool3;
  //lcpPAD<T> convpool3; /* 2nd conv & pool layer */
  Rectifier<T> relu8;
  FullyConnected<T> full9; /* hidden layer */
  Rectifier<T> relu9;
  FullyConnected<T> full10; /* hidden layer */
  Rectifier<T> relu10;
  FullyConnected<T> full11; /* output layer */
  //SoftMax output11;

  Mat3D<T> input; /*input pixel(Q 8.8)*/
  Mat3D<T> fmap1; /*feature map1(Q8.8)*/
  Mat3D<T> amap1; /*activated feature map1(Q8.8)*/
  Mat3D<T> fmap2; /*feature map1(Q8.8)*/
  Mat3D<T> pmap1; /*pooled feature map1(Q8.8)*/
  Mat3D<T> amap2; /*activated feature map1(Q8.8)*/
  Mat3D<T> fmap3; /*feature map2(Q8.8)*/
  Mat3D<T> amap3; /*activated feature map2(Q8.8)*/
  Mat3D<T> fmap4; /*feature map2(Q8.8)*/
  Mat3D<T> pmap2; /*pooled feature map2(Q8.8)*/
  Mat3D<T> amap4; /*activated feature map2(Q8.8)*/
  Mat3D<T> fmap5; /*feature map2(Q8.8)*/
  Mat3D<T> amap5; /*activated feature map2(Q8.8)*/
  Mat3D<T> fmap6; /*feature map2(Q8.8)*/
  Mat3D<T> amap6; /*activated feature map2(Q8.8)*/
  Mat3D<T> fmap7; /*feature map2(Q8.8)*/
  Mat3D<T> amap7; /*activated feature map2(Q8.8)*/
  Mat3D<T> fmap8; /*feature map2(Q8.8)*/
  Mat3D<T> pmap3; /*pooled feature map2(Q8.8)*/
  Mat3D<T> amap8; /*activated feature map2(Q8.8)*/
  Mat1D<T> amap8_flat; /*flat pooled map2(Q8.8)*/
  Mat1D<T> hunit1; /*hidden layer unit(Q8.8)*/
  Mat1D<T> aunit1; /*activated layer unit(Q8.8)*/
  Mat1D<T> hunit2; /*hidden layer unit(Q8.8)*/
  Mat1D<T> aunit2; /*activated layer unit(Q8.8)*/
  Mat1D<T> output; /*output(Q8.8)*/

public:
  Deeper();
  ~Deeper();

  void Load(string path);
  int calc(char *data, int which, int amount);

};

#include "deeper_cifar.cpp"
#endif
