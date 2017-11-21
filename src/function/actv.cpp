#ifdef _ACTV_HPP_

#include <limits>

/*activation by hinge function*/
template <typename T>
void relu(Mat3D<T>& input, Mat3D<T>& output)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  for (int n = 0; n < n_in; ++n)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        if (input[n][i][j] < 0)
          output[n][i][j] = 0;
        else
          output[n][i][j] = input[n][i][j];
}

template <typename T>
void relu(Mat1D<T>& input, Mat1D<T>& output)
{
  const int n_in = input.size();

  for (int n = 0; n < n_in; ++n)
    if (input[n] < 0)
      output[n] = 0;
    else
      output[n] = input[n];
}

int softmax(Mat1D<float>& output, int len) {
  float expsum = 0.0;

  for (int i = 0; i < len; i++)
    expsum += exp(output[i]);

  if (std::abs(expsum-0.0) < std::numeric_limits<float>::min())
    throw "softmax calculation failed";

  for (int i = 0; i < len; i++)
    output[i] = exp(output[i])/expsum;

  return 0;
}

#endif
