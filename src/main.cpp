#define _EAGER

#include "dataset.hpp"

int main(int argc, char **argv)
{
  MNIST target;
  // CIFAR10 target;
  // ImageNet target;

  target.test();

  return 0;
}

