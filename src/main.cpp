#include "fixednets.hpp"

int main(int argc, char **argv)
{
  setbuf(stdout, NULL);
  printf("\033[2J");
  // MNIST target;
  // CIFAR10 target;
  // ImageNet target;
  KITTI target;

  // target.train();
  target.test();

  return 0;
}

