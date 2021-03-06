#ifndef _BATCHNORMALIZATION_HPP_
#define _BATCHNORMALIZATION_HPP_

#include <string>

#include "matrix.hpp"

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

  void load(std::string path);
  void save(std::string path);

  void forward(Mat1D<T>& output, Mat1D<T>& input);
  void forward(Mat3D<T>& output, Mat3D<T>& input);
  void backward(Mat1D<T>& input, Mat1D<T>& output);
  void backward(Mat3D<T>& input, Mat3D<T>& output);

  void update();
};

#include "BatchNormalization.cpp"
#endif
