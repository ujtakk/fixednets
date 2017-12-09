#ifdef _FULLYCONNECTED_HPP_

#include "matrix.hpp"

template <typename T>
FullyConnected<T>::FullyConnected(int n_out, int n_in, bool quantized)
  : shape{n_out, n_in}
  , quantized(quantized)
{
  iw = zeros<T>(n_out, n_in);
  gw = zeros<T>(n_out, n_in);
  ib = zeros<T>(n_out);
  gb = zeros<T>(n_out);
}

template <typename T>
FullyConnected<T>::~FullyConnected()
{
}

template <typename T>
void FullyConnected<T>::load(std::string path)
{
  if (quantized) {
    load_quantized(iw, path, "W.txt");
    load_quantized(ib, path, "b.txt");
  }
  else {
    load_txt(iw, path+"/W.txt");
    load_txt(ib, path+"/b.txt");
  }
}

template <typename T>
void FullyConnected<T>::save(std::string path)
{
}

template <typename T>
void FullyConnected<T>::forward(Mat1D<T>& output, Mat1D<T>& input)
{
  const int n_out = shape[0];

  auto fulled = zeros<T>(n_out);
  output = zeros<T>(n_out);

  full(fulled, input, iw);
  bias(output, fulled, ib);
}

template <typename T>
void FullyConnected<T>::backward(Mat1D<T>& input, Mat1D<T>& output)
{
  T pro;
  T sum = 0;

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[1]; i++) {
    for (int j = 0; j < shape[0]; j++) {
      gw[j][i] = input[i] * output[j];
      pro = iw[j][i] * output[j];
      sum += pro;
    }

    input[i] = sum;
    sum = 0;
  }

  for (int i = 0; i < shape[0]; i++) {
    gb[i] = output[i];
  }
}

// TODO: introduce batch
template <typename T>
void FullyConnected<T>::update()
{
  float alpha = 0.001;

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; ++i)
    for (int j = 0; j < shape[1]; ++j)
      iw[i][j] -= alpha * gw[i][j];

  for (int i = 0; i < shape[0]; i++) {
    ib[i] -= alpha * gb[i];
  }
}

#endif
