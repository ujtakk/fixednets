#ifndef _FIXED_POINT_HPP_
#define _FIXED_POINT_HPP_

const int Q_BITS = 8;

template <typename T>
T Q_OFFSET = static_cast<T>(pow(2, Q_BITS));

using Q_TYPE = int16_t;

#endif
