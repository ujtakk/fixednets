#ifndef _MLP_HPP_
#define _MLP_HPP_

#include "base.hpp"
#include "layer.hpp"

template <typename T>
class MLP : Network<T, int>
{
private:
  const int N_IN    = 784;
  const int N_UNITS = 1000;
  const int N_OUT   = 10;
  const int IMWID   = 28;
  const int IMHEI   = 28;

  FullyConnected<T> full1;
  FullyConnected<T> full2;
  FullyConnected<T> full3;
  Rectifier<T> relu1;
  Rectifier<T> relu2;
  SoftMax<T> prob3;

  Mat3D<T> input;
  Mat1D<T> input_flat;
  Mat1D<T> unit1;
  Mat1D<T> bunit1;
  Mat1D<T> aunit1;
  Mat1D<T> unit2;
  Mat1D<T> bunit2;
  Mat1D<T> aunit2;
  Mat1D<T> unit3;
  Mat1D<T> output;

public:
  MLP();
  ~MLP();

  void Load(string path);
  void Save(string path);

  void Forward(string data);
  void Backward(int label);
  void Update();

  int calc(string data);
};

#include "mlp.cpp"
#endif
