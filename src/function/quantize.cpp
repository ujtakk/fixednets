#ifdef _QUANTIZE_HPP_

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
}

// QuantizeV2Op (MIN_COMBINED)
auto quantize(float x, Q_RANGE xs_range) -> quant
{
// This old version
#if 1
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
#else
#endif
}

// DequantizeOp (MIN_COMBINED)
auto dequantize(quant x, Q_RANGE xs_range) -> float
{
// This old version
#if 0
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

  // const float adjust_coef = steps / (steps - 1.0);
  // const float range = (xs_max - xs_min) * adjust_coef;
  // const float scale = range / steps;

  const int64_t quant_min =
    static_cast<int64_t>(std::numeric_limits<quant>::min());

  const double x_offset = static_cast<double>(x) - quant_min;
  const double xs_min_rounded =
    rint(xs_min / static_cast<float>(scale)) * static_cast<float>(scale);
  const double result = xs_min_rounded + (x_offset * scale);

  // const float x_offset = static_cast<float>(x) - quant_min;
  // const float xs_min_rounded =
  //   rint(xs_min / static_cast<float>(scale)) * static_cast<float>(scale);
  // const float result = xs_min_rounded + (x_offset * scale);

  return static_cast<float>(result);
#else
  const float xs_min = xs_range.first;
  const float xs_max = xs_range.second;

  // const float half_range_ = !std::is_signed<T>::value
  //                   ? 0.0f
  //                   : (static_cast<float>(std::numeric_limits<T>::max()) -
  //                      std::numeric_limits<T>::min() + 1) /
  //                         2.0f;
  const float half_range_ = 0.0f;
  // cf.) 255 = numeric_limits<quant>::max() - numeric_limits<quant>::min()
  const float scale = (xs_max - xs_min)
                    / (static_cast<float>(std::numeric_limits<quant>::max())
                                        - std::numeric_limits<quant>::min());

  // float* out_ptr = output->flat<float>().data();
  // const T* in_ptr = input.flat<T>().data();
  //
  // const int64 num_elements = input.NumElements();
  // for (int i = 0; i < num_elements; ++i) {
  //   out_ptr[i] =
  //       ((static_cast<int>(in_ptr[i]) + half_range_) * scale) +
  //       min_range;
  // }

  const float y = ((static_cast<int>(x) + half_range_) * scale) + xs_min;

  return y;
#endif
}

template <typename T>
void load_quantized(Mat1D<T>& x, std::string path, std::string name)
{
  const int size0 = x.size();

  T xs_min, xs_max;
  auto xs = zeros<quant>(size0);
  load_txt(xs_min, path+"/min_"+name);
  load_txt(xs_max, path+"/max_"+name);
  load_txt(xs, path+"/"+name);

  auto xs_range = std::make_pair(xs_min, xs_max);
  for (int i = 0; i < size0; ++i) {
    float xs_i = dequantize(xs[i], xs_range);
    x[i] = T_of_float(xs_i);
    // std::cout << (int)xs[i] << ", " << xs_i << std::endl;
  }
}

template <typename T>
void load_quantized(Mat2D<T>& x, std::string path, std::string name)
{
  const int size0 = x.size();
  const int size1 = x[0].size();

  T xs_min, xs_max;
  auto xs = zeros<quant>(size0, size1);
  load_txt(xs_min, path+"/min_"+name);
  load_txt(xs_max, path+"/max_"+name);
  load_txt(xs, path+"/"+name);

  auto xs_range = std::make_pair(xs_min, xs_max);
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      float xs_ij = dequantize(xs[i][j], xs_range);
      x[i][j] = T_of_float(xs_ij);
    }
  }
}

template <typename T>
void load_quantized(Mat3D<T>& x, std::string path, std::string name)
{
  const int size0 = x.size();
  const int size1 = x[0].size();
  const int size2 = x[0][0].size();

  T xs_min, xs_max;
  auto xs = zeros<quant>(size0, size1, size2);
  load_txt(xs_min, path+"/min_"+name);
  load_txt(xs_max, path+"/max_"+name);
  load_txt(xs, path+"/"+name);

  auto xs_range = std::make_pair(xs_min, xs_max);
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      for (int k = 0; k < size2; ++k) {
        float xs_ijk = dequantize(xs[i][j][k], xs_range);
        x[i][j][k] = T_of_float(xs_ijk);
      }
    }
  }
}

template <typename T>
void load_quantized(Mat4D<T>& x, std::string path, std::string name)
{
  const int size0 = x.size();
  const int size1 = x[0].size();
  const int size2 = x[0][0].size();
  const int size3 = x[0][0][0].size();

  T xs_min, xs_max;
  auto xs = zeros<quant>(size0, size1, size2, size3);
  load_txt(xs_min, path+"/min_"+name);
  load_txt(xs_max, path+"/max_"+name);
  load_txt(xs, path+"/"+name);

  auto xs_range = std::make_pair(xs_min, xs_max);
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      for (int k = 0; k < size2; ++k) {
        for (int l = 0; l < size3; ++l) {
          float xs_ijkl = dequantize(xs[i][j][k][l], xs_range);
          x[i][j][k][l] = T_of_float(xs_ijkl);
          // std::cout << (int)xs[i][j][k][l] << ", " << xs_ijkl << std::endl;
        }
      }
    }
  }
}

#endif
