#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <cmath>
#include <cstdint>

// using fixed = int16_t;
using fixed = int32_t;

#if 1
// const int Q_BITS = 8;
const int Q_BITS = 16;
using     Q_TYPE = fixed;

#else
using     Q_TYPE = float;
const int Q_BITS = 0;
#endif

template <typename T>
// T Q_OFFSET = static_cast<T>(pow(2, Q_BITS));
T Q_OFFSET = static_cast<T>(1 << Q_BITS);

#endif
