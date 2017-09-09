#ifndef _LAZY_HPP_
#define _LAZY_HPP_

#include "matrix.hpp"

void lazy(
  Mat3D<int> &input,
  Mat4D<int> &iw,
  Mat1D<int> &ib,
  Mat3D<int> &output,
  const int out_c, const int in_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
);

void appconv(
  Mat3D<int> &input,
  Mat4D<int> &iw,
  Mat3D<int> &out_trunc,
  const int ihei, const int iwid,
  const int out_c, const int in_c,
  const int fhei, const int fwid
);

void maxindex(
  Mat3D<int> &out_trunc,
  Mat4D<int> &index,
  const int ihei, const int iwid,
  const int out_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
);

void preconv(
  Mat4D<int> &index,
  Mat3D<int> &input,
  Mat4D<int> &iw,
  Mat1D<int> &ib,
  Mat3D<int> &output,
  const int ihei, const int iwid,
  const int out_c, const int in_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
);

#include "lazy.cpp"
#endif
