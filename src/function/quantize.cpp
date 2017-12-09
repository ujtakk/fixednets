#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "load.hpp"
#include "utility.hpp"

auto range(std::vector<float> xs) -> Q_RANGE
{
  float xs_min = *std::min_element(xs.begin(), xs.end());
  float xs_max = *std::max_element(xs.begin(), xs.end());

  return std::make_pair(xs_min, xs_max);
}

static auto to_quant_impl(float x, float xs_min, float xs_max) -> int64_t
{
  const int64_t quant_min = static_cast<double>(std::numeric_limits<quant>::min());

  if (abs(xs_min - xs_max) < std::numeric_limits<float>::epsilon()) {
    return quant_min;
  }

  const int bits = sizeof(quant) * 8;
  const int64_t steps = static_cast<int64_t>(1) << bits;
  const double adjust_coef = steps / (steps - 1.0);
  const double range = (xs_max - xs_min) * adjust_coef;
  const double scale = steps / range;

  int64_t quantized = rint(x * scale) - rint(xs_min * scale);

  quantized += quant_min;

  return quantized;
};

}

auto to_quant(float x, Q_RANGE xs_range) -> quant
{
  const float xs_min = xs_range.first;
  const float xs_max = xs_range.second;

  auto clip = [](auto x, auto xs_min, auto xs_max) {
    x = std::max(x, xs_min);
    x = std::min(x, xs_max);
    return x;
  };

  int64_t quantized = to_quant_impl(x, xs_min, xs_max);
  int64_t quant_min = std::numeric_limits<quant>::min();
  int64_t quant_max = std::numeric_limits<quant>::max();

  auto clipped = clip(quantized, quant_min, quant_max);

  return static_cast<quant>(clipped);
}

auto to_float(quant x, Q_RANGE xs_range) -> float
{
  const float xs_min = xs_range.first;
  const float xs_max = xs_range.second;

  if (abs(xs_min - xs_max) < std::numeric_limits<float>::epsilon()) {
    return xs_min;
  }

  const int bits = sizeof(quant) * 8;
  const int64_t steps = static_cast<int64_t>(1) << bits;
  const double adjust_coef = steps / (steps - 1.0);
  const double range = (xs_max - xs_min) * adjust_coef;
  const double scale = range / steps;

  const int64_t quant_min =
    static_cast<double>(std::numeric_limits<quant>::min());

  const double x_offset = static_cast<double>(x) - quant_min;
  const double xs_min_rounded =
    rint(xs_min / static_cast<float>(scale)) * static_cast<float>(scale);
  const double result = xs_min_rounded + (x_offset * scale);

  return static_cast<float>(result);
}

template <typename T>
void load_quantized(Mat1D<T>& x, std::string path, std::string name)
{
  const int len = x.size();
  auto xs = zeros<quant>(len);
  T xs_min, xs_max;
  load_txt(xs_min, path+"/min_"+name);
  load_txt(xs_max, path+"/max_"+name);
  load_txt(xs, path+name);

  auto xs_range = std::make_pair(xs_min, xs_max);
  for (int i = 0; i < len; ++i)
    x[i] = T_of_float(to_float(xs[i], xs_range));
}

