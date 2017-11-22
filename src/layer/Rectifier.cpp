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
void Rectifier<T>::forward(Mat3D<T>& input, Mat3D<T>& output)
{
  relu(input, output);
}

template <typename T>
void Rectifier<T>::forward(Mat1D<T>& input, Mat1D<T>& output)
{
  relu(input, output);
}

template <typename T>
void Rectifier<T>::backward(Mat3D<T>& output, Mat3D<T>& input)
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
void Rectifier<T>::backward(Mat1D<T>& output, Mat1D<T>& input)
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

template <typename T>
Mat3D<T> Rectifier<T>::forward(Mat3D<T>& input)
{
  auto output = relu(input);

  return output;
}

template <typename T>
Mat1D<T> Rectifier<T>::forward(Mat1D<T>& input)
{
  auto output = relu(input);

  return output;
}

template <typename T>
Mat3D<T> Rectifier<T>::backward(Mat3D<T>& output)
{
  const int onum = output.size();
  const int ohei = output[0].size();
  const int owid = output[0][0].size();
  auto input = zeros<T>(onum, ohei, owid);

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

  return input;
}

template <typename T>
Mat1D<T> Rectifier<T>::backward(Mat1D<T>& output)
{
  const int olen = output.size();
  auto input = zeros<T>(olen);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < olen; i++) {
    if (input[i] < 0)
      input[i] = 0;
    else
      input[i] = output[i];
  }

  return input;
}

#endif
