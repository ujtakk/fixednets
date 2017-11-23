#ifndef _MNIST_HPP_
#define _MNIST_HPP_

#include "network.hpp"

class ImageNet : Dataset
{
private:
  VGG<Q_TYPE>  model;
  const std::string base = "/home/work/takau/2.mlearn/imagenet_data/input224/";

  auto data = [](int label, int sample) {
    return base + std::to_string(label) + "/data" + std::to_string(sample) + ".txt";
  };

public:
  ImageNet();
  ~ImageNet();

  int predict(int label, int sample);
  void test();

  const int CLASS = 1000;
  const int SAMPLE = 100;
};

#include "imagenet.cpp"
#endif
