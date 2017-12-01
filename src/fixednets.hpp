#ifndef _FIXEDNETS_HPP_
#define _FIXEDNETS_HPP_

#define DEBUG
#ifdef DEBUG
#include <iostream>
#include <libgen.h>
#define _(s) \
  printf("line %d, file %s: %s\n", \
         __LINE__, basename(const_cast<char *>(__FILE__)), (s))
#define f(fun) do { \
  printf("line %d, file %s: %s\n", \
         __LINE__, basename(const_cast<char *>(__FILE__)), #fun); \
  (fun); \
} while (0)
#endif

#include "types.hpp"
#include "base.hpp"
#include "matrix.hpp"
#include "function.hpp"
#include "layer.hpp"
#include "network.hpp"
#include "dataset.hpp"

#endif
