#include <iostream>
#include <chrono>

#include "matrix.hpp"
#include "function/convolution.hpp"

bool test_conv_plus_pad()
{
  return true;
}

bool test_conv_aligned()
{
  using namespace std::chrono;
  system_clock::time_point start, end;
  auto show = [](auto start, auto end) {
    return duration_cast<milliseconds>(end - start).count();
  };

  auto x = zeros<float>(3, 240, 320);
  auto w = zeros<float>(64, 3, 3, 3);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 240; ++j)
      for (int k = 0; k < 320; ++k)
        x[i][j][k] = (240*320*i + 320*j + k)*0.001;

  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          w[i][j][k][l] = (3*3*3*i + 3*3*j + 3*k + l) * 0.001;

  auto y_true = zeros<float>(64, 240, 320);
  auto y_aligned = zeros<float>(64, 240, 320);

  start = system_clock::now();
  conv_plus_pad(y_true, x, w, 1, 1);
  end = system_clock::now();
  std::cout << "conv_plus_pad: " << show(start, end) << " [ms]" << std::endl;

  start = system_clock::now();
  conv_aligned(y_aligned, x, w, 1, 1);
  end = system_clock::now();
  std::cout << "conv_aligned: " << show(start, end) << " [ms]" << std::endl;

  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 240; ++j)
      for (int k = 0; k < 320; ++k)
        if (y_aligned[i][j][k] != y_true[i][j][k]) {
          std::cout << i << ", " << j << ", " << k << std::endl;
          std::cout << y_aligned[i][j][k] << ", " << y_true[i][j][k] << std::endl;
          return false;
        }

  return true;
}

int main(void)
{
  assert(test_conv_plus_pad());
  assert(test_conv_aligned());

  return 0;
}
