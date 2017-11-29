#ifndef _SQUEEZE_DET_HPP_
#define _SQUEEZE_DET_HPP_

#include <string>

#include "base.hpp"
#include "layer.hpp"

template <typename T>
struct BBoxMask
{
  Mat2D<T> det_boxes;
  Mat1D<T> det_probs;
  Mat1D<int> det_class;
};

struct DetConfig
{
  int H, W, B;
  int ANCHOR_PER_GRID;
  int CLASSES;
  int IMAGE_HEIGHT;
  int IMAGE_WIDTH;
  float PLOT_PROB_THRESH;
  float NMS_THRESH;
  float PROB_THRESH;
  int TOP_N_DETECTION;

  bool DATA_AUGMENTATION;
  int DRIFT_X;
  int DRIFT_Y;
  bool EXCLUDE_HARD_EXAMPLES;

  Mat2D<float> ANCHOR_BOX;
  int ANCHORS;
};

template <typename T>
class SqueezeDet : Network<T, BBoxMask<T>>
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

  auto merge_box_delta(Mat2D<T>& base, Mat2D<T>& delta);
  Mat1D<float> safe_exp(Mat1D<float>& x, float thresh);
  BBoxMask<T> interpret(Mat3D<T> preds);

  const int ANCHORS;
  const int ANCHOR_PER_GRID;
  const int CLASSES = 3;
  const int IMAGE_WIDTH  = 1248;
  const int IMAGE_HEIGHT = 384;
  Mat2D<T> ANCHOR_BOX;

public:
  SqueezeDet(DetConfig& conf);
  ~SqueezeDet();

  void Load(std::string path);
  void Save(std::string path);

  void Forward(std::string data);
  void Backward(int label);
  void Update();

  BBoxMask<T> calc(std::string data);
};

#include "squeeze_det.cpp"
#endif
