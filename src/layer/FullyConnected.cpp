#ifdef _FULLYCONNECTED_HPP_

#include "matrix.hpp"

using std::to_string;

template <typename T>
FullyConnected<T>::FullyConnected(int n_out, int n_in)
  : shape{n_out, n_in}
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
void FullyConnected<T>::load(string path)
{
#if 1
  load_txt(iw, path+"/W.txt");
  load_txt(ib, path+"/b.txt");
#else
  std::vector<string> filename(shape[0]);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; i++) {
    filename[i] = path+"/data"+to_string(i)+".txt";
    load_data_1d(filename[i], iw[i], ib[i], shape[1]);
  }
#endif
}

template <typename T>
void FullyConnected<T>::save(string path)
{
}

template <typename T>
void FullyConnected<T>::forward(Mat1D<T>& input, Mat1D<T>& output)
{
  const int n_out = output.size();

  Mat1D<T> fulled = zeros<T>(n_out);

  full(input, iw, fulled);
  bias(fulled, ib, output);
}

template <typename T>
void FullyConnected<T>::backward(Mat1D<T>& output, Mat1D<T>& input)
{
  int pro;
  int sum = 0;

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
// void FullyConnected<T>::update(float alpha)
void FullyConnected<T>::update()
{
  float alpha = 0.0;

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
