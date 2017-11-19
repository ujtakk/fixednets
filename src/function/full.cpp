#ifdef _FULL_HPP_

template <typename T>
void full(Mat1D<T>& input, Mat2D<T>& weight, Mat1D<T>& output)
{
  const int n_out = output.size();
  const int n_in = input.size();

  for (int i = 0; i < n_out; ++i)
    for (int j = 0; j < n_in; ++j)
      output[i] += mult_fixed(weight[i][j], input[j]);
}

#endif
