#ifdef _INTERPRET_HPP_

#include <numeric>

template <typename T>
int classify(Mat1D<T> probs)
{
  const int len = probs.size();

  int number = -1;
  T temp = std::numeric_limits<T>::min();
  for (int i = 0; i < len; ++i) {
    if (temp < probs[i]) {
      temp = probs[i];
      number = i;
    }
  }

  return number;
}

template <typename T>
std::vector<int> classify_top(Mat1D<T> probs, int range)
{
  const int len = probs.size();

  std::vector<int> number(range, -1);
  std::vector<int> index(len);
  iota(index.begin(), index.end(), 0);
  sort(index.begin(), index.end(), [&](int a, int b) {
    return probs[a] > probs[b];
  });
  for (int i = 0; i < range; ++i)
    number[i] = index[i];

  return number;
}

#endif
