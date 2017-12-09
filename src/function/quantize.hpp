#ifndef _QUANTIZE_HPP_
#define _QUANTIZE_HPP_

#include "types.hpp"
#include "matrix.hpp"

Q_RANGE range(std::vector<float> xs);

quant to_quant(float x, Q_RANGE xs_range);
float to_float(quant x, Q_RANGE xs_range);

template <typename T>
void load_quantized(Mat1D<T>& x, std::string path, std::string name);

template <typename T>
void load_quantized(Mat2D<T>& x, std::string path, std::string name);

template <typename T>
void load_quantized(Mat3D<T>& x, std::string path, std::string name);

template <typename T>
void load_quantized(Mat4D<T>& x, std::string path, std::string name);

#include "quantize.cpp"
#endif
