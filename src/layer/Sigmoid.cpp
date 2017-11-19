#ifdef _SIGMOID_HPP_

template <typename T>
Sigmoid<T>::Sigmoid()
{
}

template <typename T>
Sigmoid<T>::~Sigmoid()
{
}

template <typename T>
void Sigmoid<T>::forward(Mat1D<T>& input, Mat1D<T>& output)
{
  const int ilen = input.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < ilen; i++) {
    output[i] = (1.0/(1.0 + exp(-input[i]))) * Q_OFFSET<T>;
  }
}

template <typename T>
void Sigmoid<T>::backward(Mat1D<T>& output, Mat1D<T>& input)
{
  const int olen = output.size();

  int temp;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < olen; i++) {
    temp = (1.0/(1.0 + exp(-input[i]))) * Q_OFFSET<T>;
    input[i] = (temp * (1 - temp)) * output[i];
  }
}

template <typename T>
void Sigmoid<T>::errors(Mat1D<T> input, Mat1D<T> output, int label)
{
  const int ilen = input.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < ilen; i++) {
    if (i == label)
      output[i] = input[i] - 256;
    else
      output[i] = input[i];
  }
}

#endif
