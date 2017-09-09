#ifdef _LOAD_HPP_

#include <string>
#include <fstream>

#include "fixed_point.hpp"

using std::ifstream;
using std::ofstream;
using std::string;

void load_mnist(
  Mat2D<double> &train_set_x, Mat1D<int> &train_set_y,
  Mat2D<double> &valid_set_x, Mat1D<int> &valid_set_y,
  Mat2D<double> &test_set_x,  Mat1D<int> &test_set_y
)
{
  string filename = "mnist.dat";
  ifstream ifs(filename);

  const int n_train = 50000;
  const int n_valid = 10000;
  const int n_test = 10000;
  const int image_size = 784; // = 28 * 28

  for (int i = 0; i < n_train; i++) {
    for (int j = 0; j < image_size; j++)
      ifs >> train_set_x[i][j];
    ifs >> train_set_y[i];
  }

  for (int i = 0; i < n_valid; i++) {
    for (int j = 0; j < image_size; j++)
      ifs >> valid_set_x[i][j];
    ifs >> valid_set_y[i];
  }

  for (int i = 0; i < n_test; i++) {
    for (int j = 0; j < image_size; j++)
      ifs >> test_set_x[i][j];
    ifs >> test_set_y[i];
  }
}

/*read data file and convert to Q8.8*/
void load_convert(
  Mat2D<double> &d, Mat2D<int> &li,
  const int height, const int width
)
{
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      li[i][j] = rint(d[i][j] * Q_OFFSET<double>);
}

/*read data file and convert to Q8.8*/
template <typename T>
void load_image(string filename, Mat3D<T> &image)
{
  ifstream ifs(filename);

  const int color   = image.size();
  const int height  = image[0].size();
  const int width   = image[0][0].size();

  double val;

  for (int n = 0; n < color; n++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        ifs >> val;
        image[n][i][j] = rint(val * Q_OFFSET<double>);
      }
    }
  }
}

template <typename T>
void load_image(string filename, Mat3D<T> &li, const int height, const int width)
{
  ifstream ifs(filename);
  double d;

  for (int n = 0; n < 1; n++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        ifs >> d;
        li[n][i][j] = rint(d * Q_OFFSET<double>);
      }
    }
  }
}

template <typename T>
void load_color(string filename, Mat3D<T> &li, const int height, const int width)
{
  ifstream ifs(filename);
  double d;

  for (int n = 0; n < 3; n++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        ifs >> d;
        li[n][i][j] = rint(d * Q_OFFSET<double>);
      }
    }
  }
}


template <typename T>
void load_data(
  string filename,
  Mat2D<T> &li1,
  T &li2,
  const int height, const int width
  )
{
  ifstream ifs(filename);
  double d1, d2;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ifs >> d1;
      li1[i][j] = rint(d1 * Q_OFFSET<double>);
    }
  }
  ifs >> d2;
  li2 = rint(d2 * Q_OFFSET<double>);
}

template <typename T>
void load_data_bi(string filename, Mat2D<T> &li1, T &li2,
  const int height, const int width
)
{
  ifstream ifs(filename);
  double d1, d2;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ifs >> d1;
      if (d1 >= 0)
        d1 = 1;
      else
        d1 = -1;
      li1[i][j] = rint(d1);
    }
  }
  ifs >> d2;
  li2 = rint(d2 * Q_OFFSET<double>);
}

template <typename T>
void load_w(string filename, Mat2D<T> &li1, const int height, const int width)
{
  ifstream ifs(filename);
  double d1;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ifs >> d1;
      li1[i][j] = rint(d1 * Q_OFFSET<double>);
    }
  }
}

template <typename T>
void load_w_bi(string filename, Mat2D<T> &li1, const int height, const int width)
{
  ifstream ifs(filename);
  double d1;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ifs >> d1;
      if (d1 >= 0)
        d1 = 1;
      else
        d1 = -1;
      li1[i][j] = rint(d1);
    }
  }
}

template <typename T>
void load_b(string filename, T &li)
{
  ifstream ifs(filename);
  double d;

  ifs >> d;
  li = rint(d * Q_OFFSET<double>);
}

template <typename T>
void load_data_1d(
  string filename,
  Mat1D<T> &li1,
  T &li2,
  const int length
)
{
  ifstream ifs(filename);
  double d1[length];
  double d2;

  for (int i = 0; i < length; i++) {
    ifs >> d1[i];
    li1[i] = rint(d1[i] * Q_OFFSET<double>);
  }
  ifs >> d2;
  li2 = rint(d2 * Q_OFFSET<double>);
}

template <typename T>
void load_data_1d_bi(
  string filename,
  Mat1D<T> &li1,
  T &li2,
  const int length
)
{
  ifstream ifs(filename);
  double d1;
  double d2;

  for (int i = 0; i < length; i++) {
    ifs >> d1;
    if (d1 >= 0)
      d1 = 1;
    else
      d1 = -1;
    li1[i] = rint(d1);
  }
  ifs >> d2;
  li2 = rint(d2 * Q_OFFSET<double>);
}

template <typename T>
void load_bn(
  string filename,
  Mat1D<T> &vec,
  const int channels
)
{
  ifstream ifs(filename);
  double d[channels];

  for (int i = 0; i < channels; i++) {
    ifs >> d[i];
    vec[i] = rint(d[i] * Q_OFFSET<double>);
  }
}

template <typename T>
void load_eps(string filename, T &eps)
{
  ifstream ifs(filename);
  double d;

  ifs >> d;
  eps = rint(d * Q_OFFSET<double>);
}

template <typename T>
void save_fmap(string filename, Mat3D<T> &image)
{
  ofstream ofs(filename);

  const int color   = image.size();
  const int height  = image[0].size();
  const int width   = image[0][0].size();

  for (int n = 0; n < color; n++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        ofs << image[n][i][j] << std::endl;
      }
    }
  }
}

template <typename T>
void load(Mat1D<T> &x, std::string path)
{
  std::ifstream ifs(path);
  int tmp = 0;

  ifs >> std::hex;

  for (int i = 0; i < x.size(); ++i) {
    ifs >> tmp;
    x[i] = static_cast<T>(tmp);
  }
}

template <typename T>
void load(Mat2D<T> &x, std::string path)
{
  std::ifstream ifs(path);
  int tmp = 0;

  ifs >> std::hex;

  for (int i = 0; i < x.size(); ++i)
    for (int j = 0; i < x[0].size(); ++j) {
      ifs >> tmp;
      x[i][j] = static_cast<T>(tmp);
  }
}

template <typename T>
void load(Mat3D<T> &x, std::string path)
{
  std::ifstream ifs(path);
  int tmp = 0;

  ifs >> std::hex;

  for (int i = 0; i < x.size(); ++i)
    for (int j = 0; i < x[0].size(); ++j)
      for (int k = 0; i < x[0][0].size(); ++k) {
        ifs >> tmp;
        x[i][j][k] = static_cast<T>(tmp);
  }
}

template <typename T>
void load(Mat4D<T> &x, std::string path)
{
  std::ifstream ifs(path);
  int tmp = 0;

  ifs >> std::hex;

  for (int i = 0; i < x.size(); ++i)
    for (int j = 0; i < x[0].size(); ++j)
      for (int k = 0; i < x[0][0].size(); ++k)
        for (int l = 0; i < x[0][0][0].size(); ++l) {
          ifs >> tmp;
          x[i][j][k][l] = static_cast<T>(tmp);
  }
}

template <typename T>
void save(Mat1D<T> &y, std::string path)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (int i = 0; i < y.size(); ++i)
    ofs << y[i] << std::endl;
}

template <typename T>
void save(Mat2D<T> &y, std::string path)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (int i = 0; i < y.size(); ++i)
    for (int j = 0; i < y[0].size(); ++j)
      ofs << y[i][j] << std::endl;
}

template <typename T>
void save(Mat3D<T> &y, std::string path)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (int i = 0; i < y.size(); ++i)
    for (int j = 0; i < y[0].size(); ++j)
      for (int k = 0; i < y[0][0].size(); ++k)
        ofs << y[i][j][k] << std::endl;
}

template <typename T>
void save(Mat4D<T> &y, std::string path)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (int i = 0; i < y.size(); ++i)
    for (int j = 0; i < y[0].size(); ++j)
      for (int k = 0; i < y[0][0].size(); ++k)
        for (int l = 0; i < y[0][0][0].size(); ++l)
          ofs << y[i][j][k][l] << std::endl;
}

#endif
