#ifndef _FIXED_POINT_HPP_
#define _FIXED_POINT_HPP_

#include <cmath>

#if 0
using Q_TYPE = int16_t;

const int Q_BITS = 8;

template <typename T>
T Q_OFFSET = static_cast<T>(pow(2, Q_BITS));
#else
using Q_TYPE = float;

const int Q_BITS = 0;

template <typename T>
T Q_OFFSET = static_cast<T>(pow(2, Q_BITS));
#endif

#endif
