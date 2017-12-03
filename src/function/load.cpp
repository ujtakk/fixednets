#ifdef _LOAD_HPP_

#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "types.hpp"

template <typename T>
static inline void read(std::ifstream& ifs, T& x)
{
  float tmp = 0.0;
  ifs >> tmp;

  x = static_cast<T>(rint(tmp * Q_OFFSET<T>));
}

static inline void read(std::ifstream& ifs, float& x)
{
  float tmp = 0.0;
  ifs >> tmp;

  x = tmp;
}

#if 0
template <typename T>
void load_txt(Mat1D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    read<T>(ifs, x_i);
}

template <typename T>
void load_txt(Mat2D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      read<T>(ifs, x_ij);
}

template <typename T>
void load_txt(Mat3D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        read<T>(ifs, x_ijk);
}

template <typename T>
void load_txt(Mat4D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        for (auto& x_ijkl : x_ijk)
          read<T>(ifs, x_ijkl);
}
#else
template <typename T>
void load_txt(Mat1D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    read(ifs, x_i);
}

template <typename T>
void load_txt(Mat2D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      read(ifs, x_ij);
}

template <typename T>
void load_txt(Mat3D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        read(ifs, x_ijk);
}

template <typename T>
void load_txt(Mat4D<T>& x, std::string path)
{
  std::ifstream ifs(path);

  for (auto& x_i : x)
    for (auto& x_ij : x_i)
      for (auto& x_ijk : x_ij)
        for (auto& x_ijkl : x_ijk)
          read(ifs, x_ijkl);
}
#endif

template <typename T>
void save_txt(std::string path, Mat1D<T>& y)
{
  std::ofstream ofs(path);

  /* ofs << std::hex; */

  for (auto& y_i : y)
    ofs << y_i << std::endl;
}

template <typename T>
void save_txt(std::string path, Mat2D<T>& y)
{
  std::ofstream ofs(path);

  /* ofs << std::hex; */

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      ofs << y_ij << std::endl;
}

template <typename T>
void save_txt(std::string path, Mat3D<T>& y)
{
  std::ofstream ofs(path);

  /* ofs << std::hex; */

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      for (auto& y_ijk : y_ij)
        ofs << y_ijk << std::endl;
}

template <typename T>
void save_txt(std::string path, Mat4D<T>& y)
{
  std::ofstream ofs(path);

  /* ofs << std::hex; */

  for (auto& y_i : y)
    for (auto& y_ij : y_i)
      for (auto& y_ijk : y_ij)
        for (auto& y_ijkl : y_ijk)
          ofs << y_ijkl << std::endl;
}

// load values in range 0 ~ 255
template <typename T>
std::array<float, 2> load_img(Mat3D<T>& x, std::string path)
{
  const int in_c = x.size();
  const int n_row = x[0].size();
  const int n_col = x[0][0].size();

  cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

  std::array<float, 2> scales =
    {{(float)n_row/img.rows, (float)n_col/img.cols}};
  assert (img.channels() == in_c);

  cv::Mat img_f;
  img.convertTo(img_f, CV_32FC3);

  cv::resize(img_f, img_f, cv::Size(n_col, n_row));

  float BGR_MEANS[3] = {103.939, 116.779, 123.68};

  for (int i = 0; i < n_row; ++i) {
    for (int j = 0; j < n_col; ++j) {
      for (int k = 0; k < in_c; ++k) {
        float acc = img_f.at<cv::Vec3f>(i, j)[k] - BGR_MEANS[k];
        x[k][i][j] = T_of_float<T>(acc);
      }
    }
  }

  return scales;
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
