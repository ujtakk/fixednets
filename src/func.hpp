#ifndef _FUNC_HPP_
#define _FUNC_HPP_

using std::vector;

int approx(int value, int bias, double prob);

void conv1(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM1,
  vector<vector<int>> &etable
);

void conv1_bias(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM1,
  int bias, double prob
);

int conv2(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  vector<vector<int>> &etable
);

int conv2_bias(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  int bias, double prob
);

void fm_fm_e(
  vector<vector<vector<int>>> &infm,
  vector<vector<vector<int>>> &outfm,
  vector<vector<vector<vector<int>>>> &fweight,
  const int n_in, const int n_out, const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  vector<vector<int>> &etable
);

void fm_fm_bias(
  vector<vector<vector<int>>> &infm,
  vector<vector<vector<int>>> &outfm,
  vector<vector<vector<vector<int>>>> &fweight,
  const int n_in, const int n_out, const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  int bias, double prob
);

template <typename T>
void conv(
  vector<vector<T>> &input,
  vector<vector<T>> &fweight,
  vector<vector<T>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
);

template <typename T>
void conv_plus(
  vector<vector<T>> &input,
  vector<vector<T>> &fweight,
  vector<vector<T>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
);

template <typename T>
void conv_plus_bi(
  vector<vector<T>> &input,
  vector<vector<T>> &fweight,
  vector<vector<T>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int stride, int pad
);

template <typename T>
void conv_plus_pad(
  vector<vector<T>> &input,
  vector<vector<T>> &fweight,
  vector<vector<T>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int stride, int pad
);

void conv_approx(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int which, int amount
);

void conv_tri(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
);

void fm_fm(
  vector<vector<vector<int>>> &infm,
  vector<vector<vector<int>>> &outfm,
  vector<vector<vector<vector<int>>>> &fweight,
  const int n_in, const int n_out,
  const int ihei, const int iwid,
  const int fhei, const int fwid
);

void fm_fm_approx(
  vector<vector<vector<int>>> &infm,
  vector<vector<vector<int>>> &outfm,
  vector<vector<vector<vector<int>>>> &fweight,
  const int n_in, const int n_out,
  const int ihei, const int iwid,
  const int fhei, const int fwid,
  int which, int amount
);

void max_pooling(
  vector<vector<int>> &fmap,
  vector<vector<int>> &pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
);

void swap(vector<int> &input,int i,int j);

void median_pooling(
  vector<vector<int>> &fmap,
  vector<vector<int>> &pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
);

void add_bias(vector<vector<int>> &input, int bias, int ihei, int iwid);

void activate(vector<vector<int>> &input, const int ihei, const int iwid);

void activate_1d(vector<int> &input, const int ilen);

template <typename T>
void flatten(
  vector<vector<vector<T>>> &matrix,
  vector<T> &array
);

template <typename T>
void flatten(
  vector<vector<vector<T>>> &matrix,
  vector<T> &array,
  const int mdep, const int mhei, const int mwid
);

template <typename T>
void reshape(
  vector<T> &array,
  vector<vector<vector<T>>> &matrix
);

template <typename T>
void reshape(
  vector<T> &array,
  vector<vector<vector<T>>> &matrix,
  const int mdep, const int mhei, const int mwid
);

void full_connect(
  vector<int> &input,
  vector<int> &output,
  vector<vector<int>> &weight,
  vector<int> &bias,
  const int ilen, const int olen
);

int softmax(vector<double> &output, int len);

double mean_1d(vector<double> vec);

void lazy(
  vector<vector<vector<int>>> &input,
  vector<vector<vector<vector<int>>>> &iw,
  vector<int> &ib,
  vector<vector<vector<int>>> &output,
  const int out_c, const int in_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
);

void appconv(
  vector<vector<vector<int>>> &input,
  vector<vector<vector<vector<int>>>> &iw,
  vector<vector<vector<int>>> &out_trunc,
  const int ihei, const int iwid,
  const int out_c, const int in_c,
  const int fhei, const int fwid
);

void maxindex(
  vector<vector<vector<int>>> &out_trunc,
  vector<vector<vector<vector<int>>>> &index,
  const int ihei, const int iwid,
  const int out_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
);

void preconv(
  vector<vector<vector<vector<int>>>> &index,
  vector<vector<vector<int>>> &input,
  vector<vector<vector<vector<int>>>> &iw,
  vector<int> &ib,
  vector<vector<vector<int>>> &output,
  const int ihei, const int iwid,
  const int out_c, const int in_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
);

#include "func.cpp"
#endif
