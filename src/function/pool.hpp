#ifndef _POOL_HPP_
#define _POOL_HPP_

#include "matrix.hpp"

void max_pooling(
  Mat2D<int> &fmap, Mat2D<int> &pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
);

void median_pooling(
  Mat2D<int> &fmap, Mat2D<int> &pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
);

#include "pool.cpp"
#endif
