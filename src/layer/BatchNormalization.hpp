#ifndef _BATCHNORMALIZATION_HPP_
#define _BATCHNORMALIZATION_HPP_

#include <string>

#include "matrix.hpp"

using std::string;

template <typename T>
class BatchNormalization : ParamLayer<Mat1D<T>>
{
private:

public:
  BatchNormalization(int channels);
  ~BatchNormalization();

  Mat1D<T> gamma;
  Mat1D<T> beta;
  Mat1D<T> mean;
  Mat1D<T> var;
  Mat1D<T> std;
  T eps;

  int shape;

  void load(string path);
  void save(string path);

  void forward(Mat1D<T> &input, Mat1D<T> &output);
  void forward(Mat3D<T> &input, Mat3D<T> &output);
  void backward();
  //void backward();

  void update();
};

#include "BatchNormalization.cpp"
#endif
