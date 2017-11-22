#ifndef _SQUEEZE_DET_HPP_
#define _SQUEEZE_DET_HPP_

#include <string>

#include "base.hpp"
#include "layer.hpp"

template <typename T>
struct BBoxMask
{
};

template <typename T>
class SqueezeDet : NetWork<T, BBoxMask<T>>
{
private:
  ConvModule<T> conv1;
  MaxPooling<T> pool1;
  FireModule<T> fire2;
  FireModule<T> fire3;
  MaxPooling<T> pool3;
  FireModule<T> fire4;
  FireModule<T> fire5;
  MaxPooling<T> pool5;
  FireModule<T> fire6;
  FireModule<T> fire7;
  FireModule<T> fire8;
  FireModule<T> fire9;
  FireModule<T> fire10;
  FireModule<T> fire11;
  ConvModule<T> conv12;

  Mat3D<T> input;
  Mat3D<T> fmap1;
  Mat3D<T> pmap1;
  Mat3D<T> fmap2;
  Mat3D<T> fmap3;
  Mat3D<T> pmap3;
  Mat3D<T> fmap4;
  Mat3D<T> fmap5;
  Mat3D<T> pmap5;
  Mat3D<T> fmap6;
  Mat3D<T> fmap7;
  Mat3D<T> fmap8;
  Mat3D<T> fmap9;
  Mat3D<T> fmap10;
  Mat3D<T> fmap11;
  Mat3D<T> fmap12;

  BBoxMask<T> interpret(Mat3D<T> preds);

public:
  SqueezeDet(DetConf config);
  ~SqueezeDet();

  void Load(std::string path);
  void Save(std::string path);

  void Forward(std::string data);
  void Backward(int label);
  void Update();

  BBoxMask<T> calc(std::string data, int which, int amount);
};

#include "squeeze_det.cpp"
#endif
