#ifndef _INTERPRET_HPP_
#define _INTERPRET_HPP_

#include "matrix.hpp"

template <typename T>
int classify(Mat1D<T> probs);

template <typename T>
std::vector<int> classify_top(Mat1D<T> probs, int range);

#include "interpret.cpp"
#endif
