#ifndef _BST_HPP_
#define _BST_HPP_

#include "base.hpp"
#include "matrix.hpp"

template <typename T>
class BST : Layer<Mat1D<T>>
{
private:

public:
  BST();
  ~BST();

  void forward(Mat3D<T>& output, Mat3D<T>& input);
  void forward(Mat1D<T>& output, Mat1D<T>& input);
  void backward(Mat3D<T>& input, Mat3D<T>& output);
  void backward(Mat1D<T>& input, Mat1D<T>& output);
};

#include "BST.cpp"
#endif
