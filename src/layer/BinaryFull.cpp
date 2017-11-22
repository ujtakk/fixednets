#ifdef _BINARYFULL_HPP_

template <typename T>
BinaryFull<T>::BinaryFull(int out_channels, int in_channels)
  : shape{out_channels, in_channels}
{
  iw = zeros<T>(out_channels, in_channels);
  gw = zeros<T>(out_channels, in_channels);
  ib = zeros<T>(out_channels);
  gb = zeros<T>(out_channels);
}

template <typename T>
BinaryFull<T>::~BinaryFull()
{
}

template <typename T>
void BinaryFull<T>::load(std::string path)
{
  std::vector<std::string> filename(shape[0]);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; i++)
  {
    filename[i] = path+"/data"+std::to_string(i)+".txt";
    load_data_1d(filename[i], iw[i], ib[i], shape[1]);
  }
}

template <typename T>
void BinaryFull<T>::save(std::string path)
{
}

template <typename T>
void BinaryFull<T>::forward(Mat1D<T>& output, Mat1D<T>& input)
{
  Mat1D<T> sum(shape[0], 0);
  Mat1D<T> pro(shape[0], 0);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      if (iw[i][j] >= 0)
        pro[i] = input[j];
      else
        pro[i] = -input[j];
      sum[i] += pro[i];
    }

    output[i] = sum[i] + ib[i];
    sum[i] = 0;
  }
}

template <typename T>
void BinaryFull<T>::backward(Mat1D<T>& input, Mat1D<T>& output)
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
void BinaryFull<T>::update()
{
}

#endif
