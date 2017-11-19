#ifndef _LOAD_HPP_
#define _LOAD_HPP_

template <typename T>
void load_txt(Mat1D<T>& x, std::string path);

template <typename T>
void load_txt(Mat2D<T>& x, std::string path);

template <typename T>
void load_txt(Mat3D<T>& x, std::string path);

template <typename T>
void load_txt(Mat4D<T>& x, std::string path);

template <typename T>
void save_txt(Mat1D<T>& y, std::string path);

template <typename T>
void save_txt(Mat2D<T>& y, std::string path);

template <typename T>
void save_txt(Mat3D<T>& y, std::string path);

template <typename T>
void save_txt(Mat4D<T>& y, std::string path);

////////////////////////////////////////////////////////////
// load functions below are deprecated.
////////////////////////////////////////////////////////////

void load_mnist(
  Mat2D<double> train_set_x, Mat1D<int> train_set_y,
  Mat2D<double> valid_set_x, Mat1D<int> valid_set_y,
  Mat2D<double> test_set_x,   Mat1D<int> test_set_y
);

void load_convert(
  Mat2D<double>& d, Mat2D<int>& li,
  const int height, const int width
);

template <typename T>
void load_image(std::string filename, Mat3D<T>& image);

template <typename T>
void load_image(std::string filename, Mat3D<T>& li, const int height, const int width);

template <typename T>
void load_color(std::string filename, Mat3D<T>& li, const int height, const int width);

template <typename T>
void load_bn(std::string filename, Mat1D<T>& vec, const int channels);

template <typename T>
void load_data(std::string filename, Mat2D<T>& li1, T& li2, const int height, const int width);

template <typename T>
void load_data_bi(std::string filename, Mat2D<T>& li1, T& li2, const int height, const int width);

template <typename T>
void load_w(std::string filename, Mat2D<T>& li1, const int height, const int width);

template <typename T>
void load_w_bi(std::string filename, Mat2D<T>& li1, const int height, const int width);

template <typename T>
void load_b(std::string filename, T& li);

template <typename T>
void load_data_1d(std::string filename, Mat1D<T>& li1, T& li2, const int length);

template <typename T>
void load_data_1d_bi(std::string filename, Mat1D<T>& li1, T& li2, const int length);

template <typename T>
void save_fmap(std::string filename, Mat3D<T>& image);

#include "load.cpp"
#endif
