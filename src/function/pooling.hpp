#ifndef _POOL_HPP_
#define _POOL_HPP_

#include "matrix.hpp"

void max_pooling(
  Mat2D<int>& fmap, Mat2D<int>& pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
);

void median_pooling(
  Mat2D<int>& fmap, Mat2D<int>& pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
);

template <typename T>
void pool_max(Mat3D<T>& output, Mat3D<T>& input,
              int fil_h, int fil_w, int stride);

#include "pooling.cpp"
#endif
