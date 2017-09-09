#ifdef _ACTV_HPP_

#include <limits>

/*activation by hinge function*/
void activate(Mat2D<int> &input, const int ihei, const int iwid)
{
  for (int i = 0; i < ihei; i++)
    for (int j = 0; j < iwid; j++)
      if (input[i][j]<0)
        input[i][j] = 0;
}

void activate_1d(Mat1D<int> &input, const int ilen)
{
  for (int i = 0; i < ilen; i++)
    if (input[i]<0)
      input[i] = 0;
}

int softmax(Mat1D<double> &output, int len) {
  double expsum = 0.0;

  for (int i = 0; i < len; i++)
    expsum += exp(output[i]);

  if (std::abs(expsum-0.0) < std::numeric_limits<double>::min())
    throw "softmax calculation failed";

  for (int i = 0; i < len; i++)
    output[i] = exp(output[i])/expsum;

  return 0;
}

#endif
