#ifdef _FULL_HPP_

/*calculation of full connect layer*/
void full_connect(
  Mat1D<int> &input, Mat1D<int> &output,
  Mat2D<int> &weight, Mat1D<int> &bias,
  const int ilen, const int olen
)
{
  int pro;
  int sum = 0;
  for (int i = 0; i < olen; i++) {
    for (int j = 0; j < ilen; j++) {
      pro = input[j] * weight[i][j] >> 8;
      sum += pro;
    }
    output[i] = sum + bias[i];
    sum = 0;
  }
}

#endif
