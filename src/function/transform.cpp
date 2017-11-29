#ifdef _TRANSFORM_HPP_

template <typename T>
void flatten(Mat1D<T>& output, Mat2D<T>& input)
{
  const int size0 = input.size();
  const int size1 = input[0].size();

  int idx = 0;
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      output[idx] = input[i][j];
      ++idx;
    }
  }
}

template <typename T>
void flatten(Mat1D<T>& output, Mat3D<T>& input)
{
  const int size0 = input.size();
  const int size1 = input[0].size();
  const int size2 = input[0][0].size();

  int idx = 0;
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      for (int k = 0; k < size2; ++k) {
        output[idx] = input[i][j][k];
        ++idx;
      }
    }
  }
}

template <typename T>
void flatten(Mat1D<T>& output, Mat4D<T>& input)
{
  const int size0 = input.size();
  const int size1 = input[0].size();
  const int size2 = input[0][0].size();
  const int size3 = input[0][0][0].size();

  int idx = 0;
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      for (int k = 0; k < size2; ++k) {
        for (int l = 0; l < size3; ++l) {
          output[idx] = input[i][j][k][l];
          ++idx;
        }
      }
    }
  }
}

template <typename T>
void reshape(Mat2D<T>& output, Mat1D<T>& input)
{
  const int size0 = output.size();
  const int size1 = output[0].size();

  int idx = 0;
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      output[i][j] = input[idx];
      ++idx;
    }
  }
}

template <typename T>
void reshape(Mat3D<T>& output, Mat1D<T>& input)
{
  const int size0 = output.size();
  const int size1 = output[0].size();
  const int size2 = output[0][0].size();

  int idx = 0;
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      for (int k = 0; k < size2; ++k) {
        output[i][j][k] = input[idx];
        ++idx;
      }
    }
  }
}

template <typename T>
void reshape(Mat4D<T>& output, Mat1D<T>& input)
{
  const int size0 = output.size();
  const int size1 = output[0].size();
  const int size2 = output[0][0].size();
  const int size3 = output[0][0][0].size();

  int idx = 0;
  for (int i = 0; i < size0; ++i) {
    for (int j = 0; j < size1; ++j) {
      for (int k = 0; k < size2; ++k) {
        for (int l = 0; l < size3; ++l) {
          output[i][j][k][l] = input[idx];
          ++idx;
        }
      }
    }
  }
}

template <typename T>
static inline int length(Mat1D<T>& x)
{
  return x.size();
}

template <typename T>
static inline int length(Mat2D<T>& x)
{
  return x.size() * x[0].size();
}

template <typename T>
static inline int length(Mat3D<T>& x)
{
  return x.size() * x[0].size() * x[0][0].size();
}

template <typename T>
static inline int length(Mat4D<T>& x)
{
  return x.size() * x[0].size() * x[0][0].size() * x[0][0][0].size();
}

template <typename T, typename MatOut, typename MatIn>
void reshape(MatOut& output, MatIn& input)
{
  auto flat = zeros<T>(length<T>(input));
  flatten(flat, input);
  reshape(output, flat);
}

template <typename Mat>
void concat(Mat& c, Mat& a, Mat& b)
{
  const int n_a = a.size();
  const int n_b = b.size();

  assert(c.size() == n_b + n_a);

  for (int i = 0; i < n_a; ++i)
    c[i] = a[i];

  for (int i = 0; i < n_b; ++i)
    c[i+n_a] = b[i];
}

#endif
