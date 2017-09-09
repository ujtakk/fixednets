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
  std::vector<string> filename(shape[0]);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; i++) {
    filename[i] = path+"/data"+to_string(i)+".txt";
    load_data_1d(filename[i], iw[i], ib[i], shape[1]);
  }
}

template <typename T>
void FullyConnected<T>::save(string path)
{
}

template <typename T>
void FullyConnected<T>::forward(Mat1D<T> &input, Mat1D<T> &output)
{
  Mat1D<T> sum(shape[0], 0);
  Mat1D<int> pro(shape[0], 0);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      pro[i] = iw[i][j] * input[j];
      if (pro[i] >= 0)
        sum[i] += pro[i] / Q_OFFSET<T>;
      else
        sum[i] += pro[i] / Q_OFFSET<T> - 1;
    }

    output[i] = sum[i] + ib[i];
    sum[i] = 0;
  }
}

template <typename T>
void FullyConnected<T>::backward(Mat1D<T> &output, Mat1D<T> &input)
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

template <typename T>
void FullyConnected<T>::update()
{
}

#endif
