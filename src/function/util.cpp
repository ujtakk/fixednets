#ifdef _UTIL_HPP_

#include <random>

/*add bias*/
void add_bias(Mat2D<int> &input,int bias,int ihei,int iwid)
{
  for (int i = 0; i < ihei; i++)
    for (int j = 0; j < iwid; j++)
      input[i][j] = input[i][j] + bias;
}

double mean_1d(Mat1D<double> vec)
{
  double a=0.0;

  for (int i = 0; i < (int)vec.size(); i++)
    a += vec[i];

  return a / (double)vec.size();
}

int approx(int value, int bias, double prob)
{
  int biased = value;
  int rnd_value;
  std::mt19937 mt(10);

  rnd_value = std::abs((int)mt()) % 10000000;

  if ((rnd_value / 10000000.0) < prob)
    biased = biased + bias;

  return biased;
}

/*flatten 3D matrix*/
template <typename T>
void flatten(Mat3D<T> &matrix, Mat1D<T> &array)
{
  const int mdep = matrix.size();
  const int mhei = matrix[0].size();
  const int mwid = matrix[0][0].size();

  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        array[i*mhei*mwid+j*mwid+k] = matrix[i][j][k];
}

template <typename T>
void flatten(Mat3D<T> &matrix, Mat1D<T> &array,
             const int mdep, const int mhei, const int mwid)
{
  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        array[i*mhei*mwid+j*mwid+k] = matrix[i][j][k];
}

/*reshape 3D matrix*/
template <typename T>
void reshape(Mat1D<T> &array, Mat3D<T> &matrix)
{
  const int mdep = matrix.size();
  const int mhei = matrix[0].size();
  const int mwid = matrix[0][0].size();

  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        matrix[i][j][k] = array[i*mhei*mwid+j*mwid+k];
}

template <typename T>
void reshape(Mat1D<T> &array, Mat3D<T> &matrix,
             const int mdep, const int mhei, const int mwid)
{
  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        matrix[i][j][k] = array[i*mhei*mwid+j*mwid+k];
}

#endif
