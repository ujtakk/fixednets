#ifdef _RECTIFIER_HPP_

template <typename T>
Rectifier<T>::Rectifier()
{
}

template <typename T>
Rectifier<T>::~Rectifier()
{
}

template <typename T>
void Rectifier<T>::forward(Mat3D<T> &input, Mat3D<T> &output)
{
  const int inum = input.size();
  const int ihei = input[0].size();
  const int iwid = input[0][0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < inum; n++) {
    for (int i = 0; i < ihei; i++) {
      for (int j = 0; j < iwid; j++) {
        if (input[n][i][j] < 0)
          output[n][i][j] = 0;
        else
          output[n][i][j] = input[n][i][j];
      }
    }
  }
}

template <typename T>
void Rectifier<T>::forward(Mat1D<T> &input, Mat1D<T> &output)
{
  const int ilen = input.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < ilen; i++) {
    if (input[i] < 0)
      output[i] = 0;
    else
      output[i] = input[i];
  }
}

template <typename T>
void Rectifier<T>::backward(Mat3D<T> &output, Mat3D<T> &input)
{
  const int onum = output.size();
  const int ohei = output[0].size();
  const int owid = output[0][0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < onum; n++) {
    for (int i = 0; i < ohei; i++) {
      for (int j = 0; j < owid; j++) {
        if (input[n][i][j] < 0)
          input[n][i][j] = 0;
        else
          input[n][i][j] = output[n][i][j];
      }
    }
  }
}

template <typename T>
void Rectifier<T>::backward(Mat1D<T> &output, Mat1D<T> &input)
{
  const int olen = output.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < olen; i++) {
    if (input[i] < 0)
      input[i] = 0;
    else
      input[i] = output[i];
  }
}

#endif
