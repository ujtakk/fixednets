#ifdef _FULL_HPP_

template <typename T>
void full(Mat1D<T>& input, Mat2D<T>& weight, Mat1D<T>& output)
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

template <typename T>
Mat1D<T> full(Mat1D<T>& input, Mat2D<T>& weight)
{
  const int n_out = weight.size();
  const int n_in = weight[0].size();
  auto output = zeros<T>(n_out);

  for (int i = 0; i < n_out; ++i) {
    output[i] = 0;
    for (int j = 0; j < n_in; ++j) {
      output[i] += mul(weight[i][j], input[j]);
    }
  }

  return output;
}

#endif
