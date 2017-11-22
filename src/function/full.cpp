#ifdef _FULL_HPP_

template <typename T>
void full(Mat1D<T>& output, Mat1D<T>& input, Mat2D<T>& weight)
{
  const int n_out = output.size();
  const int n_in = input.size();

  for (int i = 0; i < n_out; ++i) {
    output[i] = 0;
    for (int j = 0; j < n_in; ++j) {
      output[i] += mul(weight[i][j], input[j]);
    }
  }
}

#endif
