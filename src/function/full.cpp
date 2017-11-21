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
static void transpose(Mat2D<T> &A_t, Mat2D<T> &A)
{
  int row = A_t.size();
  int col = A_t[0].size();

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      A_t[i][j] = A[j][i];
    }
  }
}

template <typename T>
void matmul_t(Mat2D<T> &C, Mat2D<T> &A, Mat2D<T> &B)
{
  int row = C.size();
  int col = C[0].size();
  int elem = B.size();

  Mat2D<T> B_t = zeros<T>(col, elem);
  transpose(B_t, B);

  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      for (int k = 0; k < elem; ++k) {
        C[i][j] += A[i][k] * B_t[j][k];
      }
    }
  }
}

#endif
