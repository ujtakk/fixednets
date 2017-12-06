#ifdef _NORM_HPP_

template <typename T>
void norm_batch(Mat1D<T>& output, Mat1D<T>& input, T gamma, T beta, T eps, T mean, T std)
{
  const int ilen = input.size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i=0; i<ilen; i++) {
    output[i] = mlt(gamma[i], dvd((input[i] - mean[i]), std[i]))
              + beta[i];
  }
}

template <typename T>
void norm_batch(Mat3D<T>& output, Mat3D<T>& input, T gamma, T beta, T eps, T mean, T std)
{
  const int inum = input.size();
  const int ihei = input[0].size();
  const int iwid = input[0][0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n=0; n<inum; n++) {
    for (int i=0; i<ihei; i++) {
      for (int j=0; j<iwid; j++) {
        output[n][i][j] = mlt(gamma[n], dvd((input[n][i][j] - mean[n]), std[n]))
                        + beta[n];
      }
    }
  }
}

#endif
