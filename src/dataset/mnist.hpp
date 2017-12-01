#ifndef _MNIST_HPP_
#define _MNIST_HPP_

#include "network.hpp"

#define _LENET
// #define _MLP

class MNIST : Dataset
{
private:
#if defined _MLP
  MLP<Q_TYPE> model;
#elif defined _LENET
  LeNet<Q_TYPE> model;
#endif

  const std::string base = "../data/mnist/input/";
  std::string data(int label, int sample) {
    using std::to_string;
    return base + to_string(label) + "/data" + to_string(sample) + ".txt";
  };

public:
  MNIST();
  ~MNIST();

  void train();

  int predict(int label, int sample);
  void test();

  const int CLASS = 10;
  const int SAMPLE = 100;
};

#include "mnist.cpp"
#endif
