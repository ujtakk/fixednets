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
  auto show = [](auto name, auto start, auto end) {
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << name << ":\t" << elapsed << "\t[ms]" << std::endl;
  };

  const int n_out = 128;
  const int n_in  = 64;
  const int img_h = 240, img_w = 320;
  const int fil_h = 3, fil_w = 3;

  auto x = zeros<float>(n_in, img_h, img_w);
  auto w = zeros<float>(n_out, n_in, fil_h, fil_w);

  for (int i = 0; i < n_in; ++i)
    for (int j = 0; j < img_h; ++j)
      for (int k = 0; k < img_w; ++k)
        x[i][j][k] = (img_h*img_w*i + img_w*j + k)*0.001;

  for (int i = 0; i < n_out; ++i)
    for (int j = 0; j < n_in; ++j)
      for (int k = 0; k < fil_h; ++k)
        for (int l = 0; l < fil_w; ++l)
          w[i][j][k][l] =
            (n_in*fil_h*fil_w*i + fil_h*fil_w*j + fil_w*k + l) * 0.001;

  auto y_true = zeros<float>(n_out, img_h, img_w);
  auto y_aligned = zeros<float>(n_out, img_h, img_w);

  start = system_clock::now();
  conv_plus_pad(y_true, x, w, 1, 1);
  end = system_clock::now();
  show("conv_plus_pad", start, end);

  // TODO: image reshape with allocation is bottle-neck
  start = system_clock::now();
  conv_aligned(y_aligned, x, w, 1, 1);
  end = system_clock::now();
  show("conv_aligned", start, end);

  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 240; ++j)
      for (int k = 0; k < 320; ++k)
        if (y_aligned[i][j][k] != y_true[i][j][k])
          return false;

  return true;
}

int main(void)
{
  assert(test_conv_plus_pad());
  assert(test_conv_aligned());

  return 0;
}
