#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <cmath>
#include <cstdint>

#if 0
// using fixed = int16_t;
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

#if 0
using fixed = int16_t;
#endif

#endif
