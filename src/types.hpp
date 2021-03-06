#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <cstdint>
#include <utility>

// #define FLOAT
#define FIXED
#define QUANT

// integer(+sign) part requires 6 bits.
// #define Q8_8
// #define Q16_16
#define Q8_24

#if defined(Q8_8)
using fixed = int16_t;
#elif defined(Q8_24) || defined(Q16_16)
using fixed = int32_t;
#endif

using quant = uint8_t;

#if defined(FIXED)

#if defined(Q8_8)
const int Q_BITS = 8;
#elif defined(Q16_16)
const int Q_BITS = 16;
#elif defined(Q8_24)
const int Q_BITS = 24;
#endif
using     Q_TYPE = fixed;

#else

using     Q_TYPE = float;
const int Q_BITS = 0;

#endif

template <typename T>
const T   Q_OFFSET = static_cast<T>(1 << Q_BITS);

using     Q_RANGE = std::pair<float, float>;

#endif
