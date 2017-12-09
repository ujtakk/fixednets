#include "function/utility.hpp"

bool test_apply()
{
  auto succ = [](auto i) {
    return i + 1;
  };

  const auto len = 100;
  std::vector<int> v(len);
  for (int i = 0; i < len; ++i)
    v[i] = i;

  auto fv = apply(succ, v);
  for (int i = 0; i < len; ++i)
    if (fv[i] != v[i] + 1)
      return false;

  return true;
}

int main(void)
{
  assert(test_apply());

  return 0;
}
