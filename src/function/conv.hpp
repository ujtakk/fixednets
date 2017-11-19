#ifndef _CONV_HPP_
#define _CONV_HPP_

#include "matrix.hpp"

void conv_error1(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM1,
  Mat2D<int>& etable
);

void conv_error1_bias(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM1,
  int bias, double prob
);

int conv_error2(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  Mat2D<int>& etable
);

int conv_error2_bias(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  int bias, double prob
);

void fm_fm_e(Mat3D<int>& infm, Mat3D<int>& outfm, Mat4D<int>& fweight,
  const int n_in, const int n_out, const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  Mat2D<int>& etable
);

void fm_fm_bias(Mat3D<int>& infm, Mat3D<int>& outfm, Mat4D<int>& fweight,
  const int n_in, const int n_out, const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  int bias, double prob
);

void conv_approx(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int which, int amount
);

void conv_tri(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
);

void fm_fm(Mat3D<int>& infm, Mat3D<int>& outfm, Mat4D<int>& fweight,
  const int n_in, const int n_out,
  const int ihei, const int iwid,
  const int fhei, const int fwid
);

void fm_fm_approx(Mat3D<int>& infm, Mat3D<int>& outfm, Mat4D<int>& fweight,
  const int n_in, const int n_out,
  const int ihei, const int iwid,
  const int fhei, const int fwid,
  int which, int amount
);

template <typename T>
void conv(Mat2D<T>& input, Mat2D<T>& fweight, Mat2D<T>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
);

template <typename T>
void conv_plus(Mat2D<T>& input, Mat2D<T>& fweight, Mat2D<T>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
);

template <typename T>
void conv_plus_bi(Mat2D<T>& input, Mat2D<T>& fweight, Mat2D<T>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int stride, int pad
);

template <typename T>
void conv_plus_pad(Mat2D<T>& input, Mat2D<T>& fweight, Mat2D<T>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int stride, int pad
);

template <typename T>
void conv_plus_pad(Mat3D<T>& input, Mat4D<T>& weight, Mat3D<T>& output,
                   int stride, int pad);

#include "conv.cpp"
#endif
