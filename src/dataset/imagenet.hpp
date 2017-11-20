#ifndef _MNIST_HPP_
#define _MNIST_HPP_

#include "network.hpp"

class ImageNet : Dataset
{
private:
#if defined _ALEX
  AlexNet<Q_TYPE> model;
  const std::string base = "/home/work/takau/2.mlearn/imagenet_data/input/";
#elif defined _VGG
  VGGNet<Q_TYPE>  model;
  const std::string base = "/home/work/takau/2.mlearn/imagenet_data/input224/";
#endif

  auto data = [](int label, int sample) {
    return base + to_string(label) + "/data" + to_string(sample) + ".txt";
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
