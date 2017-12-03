#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <cmath>
#include <cstdint>

using fixed = int16_t;

#if 0
// using Q_TYPE = int16_t;
// const int Q_BITS = 8;
using Q_TYPE = int32_t;
const int Q_BITS = 16;
#else
using Q_TYPE = float;
const int Q_BITS = 0;
#endif

template <typename T>
T Q_OFFSET = static_cast<T>(pow(2, Q_BITS));

#endif
