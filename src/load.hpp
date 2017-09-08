#ifndef _LOAD_HPP_
#define _LOAD_HPP_

using std::string;

void load_mnist(
  Mat2D<double> train_set_x, Mat1D<int> train_set_y,
  Mat2D<double> valid_set_x, Mat1D<int> valid_set_y,
  Mat2D<double> test_set_x,   Mat1D<int> test_set_y
);

void load_convert(
  Mat2D<double> &d, Mat2D<int> &li,
  const int height,const int width
);

template <typename T>
void load_image(string filename, Mat3D<T> &image);

template <typename T>
void load_image(string filename, Mat3D<T> &li, const int height,const int width);

template <typename T>
void load_color(string filename, Mat3D<T> &li, const int height,const int width);

template <typename T>
void load_bn(string filename, Mat1D<T> &vec, const int channels);

template <typename T>
void load_data(string filename, Mat2D<T> &li1, T &li2, const int height, const int width);

template <typename T>
void load_data_bi(string filename, Mat2D<T> &li1, T &li2, const int height, const int width);

template <typename T>
void load_w(string filename, Mat2D<T> &li1, const int height,const int width);

template <typename T>
void load_w_bi(string filename, Mat2D<T> &li1, const int height,const int width);

template <typename T>
void load_b(string filename, T &li);

template <typename T>
void load_data_1d(string filename, Mat1D<T> &li1, T &li2, const int length);

template <typename T>
void load_data_1d_bi(string filename, Mat1D<T> &li1, T &li2, const int length);

template <typename T>
void save_fmap(string filename, Mat3D<T> &image);

template <typename T>
void load(Mat1D<T> &x, std::string path);

template <typename T>
void load(Mat2D<T> &x, std::string path);

template <typename T>
void load(Mat3D<T> &x, std::string path);

template <typename T>
void load(Mat4D<T> &x, std::string path);

template <typename T>
void save(Mat1D<T> &y, std::string path);

template <typename T>
void save(Mat2D<T> &y, std::string path);

template <typename T>
void save(Mat3D<T> &y, std::string path);

template <typename T>
void save(Mat4D<T> &y, std::string path);

#include "load.cpp"
#endif
