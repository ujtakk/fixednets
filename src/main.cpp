#include "fixednets.hpp"

int main(int argc, char **argv)
{
  // MNIST target;
  // CIFAR10 target;
  // ImageNet target;
  KITTI target;

  // target.train();
  target.test();

  return 0;
}

