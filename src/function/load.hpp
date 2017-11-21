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

template <typename T>
void load_image(std::string filename, Mat3D<T>& image);

template <typename T>
void load_data(std::string filename, Mat2D<T>& li1, T& li2, const int height, const int width);

template <typename T>
void load_w(std::string filename, Mat2D<T>& li1, const int height, const int width);

template <typename T>
void load_b(std::string filename, T& li);

template <typename T>
void load_data_1d(std::string filename, Mat1D<T>& li1, T& li2, const int length);

#include "load.cpp"
#endif
