#ifdef _LOAD_HPP_

#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "types.hpp"

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

#if 0
template <typename T>
void load_txt(Mat1D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    x_i = read<T>(ifs);
}

template <typename T>
void load_txt(Mat2D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      x_ij = read<T>(ifs);
}

template <typename T>
void load_txt(Mat3D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        x_ijk = read<T>(ifs);
}

template <typename T>
void load_txt(Mat4D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        for (auto& x_ijkl : x_ijk)
          x_ijkl = read<T>(ifs);
}
#else
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
#endif

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

// load values in range 0 ~ 255
template <typename T>
void load_img(Mat3D<T>& x, std::string path)
{
  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
  cv::resize(img, img, cv::Size(1248, 384));

  assert (img.channels() == 3);
  x = zeros<T>(img.channels(), img.rows, img.cols);

  int idx = 0;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      for (int k = 0; k < img.channels(); ++k) {
        // assert(img.data[idx] == img.at<Vec3b>(i, j)[k]);
        x[k][i][j] = img.data[idx];
        ++idx;
      }
    }
  }
}

template <typename T>
void save_img(std::string path, Mat3D<T>& x)
{
  assert (x.size() == 3);
  cv::Mat img(x[0].size(), x[0][0].size(), CV_8UC3);

  int idx = 0;
  for (int i = 0; i < img.rows; ++i) {
    for (int j = 0; j < img.cols; ++j) {
      for (int k = 0; k < img.channels(); ++k) {
        // assert(img.data[idx] == img.at<Vec3b>(i, j)[k]);
        img.data[idx] = x[k][i][j];
        ++idx;
      }
    }
  }

  cv::imwrite(path, img);
}

#endif
