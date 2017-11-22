#ifdef _LOAD_HPP_

#include <string>
#include <fstream>

#include "types.hpp"

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
Mat1D<T> load_txt(std::string path, int size0)
{
  std::ifstream ifs(path);
  auto x = zeros<T>(size0);

  for (auto& x_i : x)
    x_i = read(ifs);

  return x;
}

template <typename T>
Mat2D<T> load_txt(std::string path, int size0, int size1)
{
  std::ifstream ifs(path);
  auto x = zeros<T>(size0, size1);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      x_ij = read(ifs);

  return x;
}

template <typename T>
Mat3D<T> load_txt(std::string path, int size0, int size1, int size2)
{
  std::ifstream ifs(path);
  auto x = zeros<T>(size0, size1, size2);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        x_ijk = read(ifs);

  return x;
}

template <typename T>
Mat4D<T> load_txt(std::string path, int size0, int size1, int size2, int size3)
{
  std::ifstream ifs(path);
  auto x = zeros<T>(size0, size1, size2, size3);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        for (auto& x_ijkl : x_ijk)
          x_ijkl = read(ifs);

  return x;
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

template <typename T>
void save_txt(std::string path, Mat1D<T>& y)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (auto& y_i : y)
    ofs << y_i << std::endl;
}

template <typename T>
void save_txt(std::string path, Mat2D<T>& y)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      ofs << y_ij << std::endl;
}

template <typename T>
void save_txt(std::string path, Mat3D<T>& y)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      for (auto& y_ijk : y_ij)
        ofs << y_ijk << std::endl;
}

template <typename T>
void save_txt(std::string path, Mat4D<T>& y)
{
  std::ofstream ofs(path);

  ofs << std::hex;

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      for (auto& y_ijk : y_ij)
        for (auto& y_ijkl : y_ijk)
          ofs << y_ijkl << std::endl;
}

////////////////////////////////////////////////////////////
// load functions below are deprecated.
////////////////////////////////////////////////////////////


template <typename T>
void load_data(
  std::string filename,
  Mat2D<T>& li1,
  T& li2,
  const int height, const int width
  )
{
  std::ifstream ifs(filename);
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
void load_w(std::string filename, Mat2D<T>& li1, const int height, const int width)
{
  std::ifstream ifs(filename);
  double d1;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      ifs >> d1;
      li1[i][j] = rint(d1 * Q_OFFSET<double>);
    }
  }
}

template <typename T>
void load_b(std::string filename, T& li)
{
  std::ifstream ifs(filename);
  double d;

  ifs >> d;
  li = rint(d * Q_OFFSET<double>);
}

template <typename T>
void load_data_1d(
  std::string filename,
  Mat1D<T>& li1,
  T& li2,
  const int length
)
{
  std::ifstream ifs(filename);
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
