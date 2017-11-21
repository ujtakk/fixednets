#ifdef _LOAD_HPP_

#include <string>
#include <fstream>

#include "fixed_point.hpp"

using std::ifstream;
using std::ofstream;
using std::string;

#include <iostream>
static inline float read(std::ifstream& ifs)
{
  float tmp = 0.0;
  ifs >> tmp;

  return tmp;
}

template <typename T>
static inline T read(std::ifstream& ifs)
{
  double tmp = 0.0;
  ifs >> tmp;

  return static_cast<T>(rint(tmp * Q_OFFSET<T>));
}

template <typename T>
void load_txt(Mat1D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    x_i = read(ifs);
}

template <typename T>
void load_txt(Mat2D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      x_ij = read(ifs);
}

template <typename T>
void load_txt(Mat3D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        x_ijk = read(ifs);
}

template <typename T>
void load_txt(Mat4D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        for (auto& x_ijkl : x_ijk)
          x_ijkl = read(ifs);
}

template <typename T>
void save_txt(Mat1D<T>& y, std::string path)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (auto& y_i : y)
    ofs << y_i << std::endl;
}

template <typename T>
void save_txt(Mat2D<T>& y, std::string path)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      ofs << y_ij << std::endl;
}

template <typename T>
void save_txt(Mat3D<T>& y, std::string path)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      for (auto& y_ijk : y_ij)
        ofs << y_ijk << std::endl;
}

template <typename T>
void save_txt(Mat4D<T>& y, std::string path)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      for (auto& y_ijk : y_ij)
        for (auto& y_ijkl : y_ijk)
          ofs << y_ijkl << std::endl;
}

/*read data file and convert to Q8.8*/
template <typename T>
void load_image(string filename, Mat3D<T>& image)
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
void load_data(
  string filename,
  Mat2D<T>& li1,
  T& li2,
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
void load_w(string filename, Mat2D<T>& li1, const int height, const int width)
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
void load_b(string filename, T& li)
{
  ifstream ifs(filename);
  double d;

  ifs >> d;
  li = rint(d * Q_OFFSET<double>);
}

template <typename T>
void load_data_1d(
  string filename,
  Mat1D<T>& li1,
  T& li2,
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

#endif
