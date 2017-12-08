#ifdef _FULL_HPP_

template <typename T>
void full(Mat1D<T>& output, Mat1D<T>& input, Mat2D<T>& weight)
{
  const int n_out = weight.size();
  const int n_in = weight[0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < n_out; ++i)
    for (int j = 0; j < n_in; ++j)
      output[i] += mlt(weight[i][j], input[j]);
}

template <typename T>
void gemm(Mat2D<T>& output, Mat2D<T>& input, Mat2D<T>& weight)
{
  int batch = output.size();
  int n_out = output[0].size();
  int n_in = weight[0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < batch; ++i)
    for (int j = 0; j < n_out; ++j)
      for (int k = 0; k < n_in; ++k)
        output[i][j] += input[i][k] * weight[j][k];
}

#endif
