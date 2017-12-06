#include "fixednets.hpp"

int main(int argc, char **argv)
{
  setbuf(stdout, NULL);
  // printf("\033[2J");
  // MNIST target;
  // CIFAR10 target;
  // ImageNet target;
  KITTI target;

  // target.train();
  target.test();
  // for (int i = 0; i < 10; ++i) {
  //   float f = i / 10.0;
  //   fixed Tf = T_of_float<fixed>(f);
  //   float fTf = float_of_T(Tf);
  //   printf("%10.8f %10d %10.8f\n", f, Tf, fTf);
  // }

  return 0;
}

