#ifndef _CIFAR10_HPP_
#define _CIFAR10_HPP_

#include "network.hpp"

class CIFAR10 : Dataset
{
private:
  CIFAR<Q_TYPE> model;

  const std::string base = "/home/work/takau/2.mlearn/cifar10_data/pro_input/";
  std::string data(int label, int sample) {
    return base + to_string(label) + "/data" + to_string(sample) + ".txt";
  };

public:
  CIFAR10();
  ~CIFAR10();

  int predict(int label, int sample);
  void test();

  const int CLASS = 10;
  const int SAMPLE = 100;
};

#include "cifar10.cpp"
#endif
