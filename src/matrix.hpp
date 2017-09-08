#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <vector>

template <typename T>
using Mat1D = std::vector<T>;

template <typename T>
using Mat2D = std::vector< std::vector<T> >;

template <typename T>
using Mat3D = std::vector< std::vector< std::vector<T> > >;

template <typename T>
using Mat4D = std::vector< std::vector< std::vector< std::vector<T> > > >;

template <typename T>
Mat1D<T> zeros(int size1);

template <typename T>
Mat2D<T> zeros(int size1, int size2);

template <typename T>
Mat3D<T> zeros(int size1, int size2, int size3);

template <typename T>
Mat4D<T> zeros(int size1, int size2, int size3, int size4);

#include "matrix.cpp"
#endif
