#ifdef _FULL_HPP_

template <typename T>
void full(Mat1D<T>& output, Mat1D<T>& input, Mat2D<T>& weight)
{
  const int n_out = output.size();
  const int n_in = input.size();

  #ifdef _OPENMP
  std::cout << "openmp" << std::endl;
  #pragma omp parallel for
  #endif
  for (int i = 0; i < n_out; ++i) {
    T acc = 0;
    for (int j = 0; j < n_in; ++j) {
      acc += mul(weight[i][j], input[j]);
    }
    output[i] = acc;
  }
}

#endif
