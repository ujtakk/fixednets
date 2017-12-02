#ifndef _LOAD_HPP_
#define _LOAD_HPP_

#include "matrix.hpp"

template <typename T>
void load_txt(Mat1D<T>& x, std::string path);

template <typename T>
void load_txt(Mat2D<T>& x, std::string path);

template <typename T>
void load_txt(Mat3D<T>& x, std::string path);

template <typename T>
void load_txt(Mat4D<T>& x, std::string path);

template <typename T>
void save_txt(std::string path, Mat1D<T>& y);

template <typename T>
void save_txt(std::string path, Mat2D<T>& y);

template <typename T>
void save_txt(std::string path, Mat3D<T>& y);

template <typename T>
void save_txt(std::string path, Mat4D<T>& y);

template <typename T>
std::array<float, 2> load_img(Mat3D<T>& x, std::string path);

template <typename T>
void save_img(std::string path, Mat3D<T>& x);

#include "load.cpp"
#endif
