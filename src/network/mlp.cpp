#ifdef _MLP_HPP_

#include <limits>

template <typename T>
MLP<T>::MLP()
  : full1{N_UNITS, N_IN}
  , full2{N_OUT, N_UNITS}
{
  input       = zeros<T>(1, IMHEI, IMWID);
  input_flat  = zeros<T>(1 * IMHEI * IMWID);
  unit1       = zeros<T>(N_UNITS);
  aunit1      = zeros<T>(N_UNITS);
  unit2       = zeros<T>(N_OUT);
  output      = zeros<T>(N_OUT);
}

template <typename T>
MLP<T>::~MLP()
{
}

template <typename T>
void MLP<T>::Load(std::string path)
{
  full1.load(path+"/full1");
  full2.load(path+"/full2");
}

template <typename T>
void MLP<T>::Save(std::string path)
{
  full1.save(path+"/full1");
  full2.save(path+"/full2");
}

template <typename T>
void MLP<T>::Forward(std::string data)
{
  load_txt(input, data);
  flatten(input, input_flat);

  full1.forward(input_flat, unit1);
  relu1.forward(unit1, aunit1);
  full2.forward(aunit1, unit2);

  prob2.prob(unit2, output);
}

#include <cstdio>
template <typename T>
void MLP<T>::Backward(int label)
{
  prob2.loss(label, output, unit2);

  full2.backward(unit2, aunit1);
  relu1.backward(aunit1, unit1);
  full1.backward(unit1, input_flat);
}

template <typename T>
void MLP<T>::Update()
{
  full1.update();
  full2.update();
}

template <typename T>
int MLP<T>::calc(std::string data, int which, int amount)
{
  load_txt(input, data);
  flatten(input, input_flat);

  full1.forward(input_flat, unit1);
  relu1.forward(unit1, aunit1);
  // full2.forward(aunit1, output);
  full2.forward(aunit1, unit2);
  prob2.forward(unit2, output);

  int number = -1;
  T temp = std::numeric_limits<T>::min();
  for (int i = 0; i < N_OUT; ++i) {
    if (temp < output[i]) {
       temp = output[i];
       number = i;
    }
  }

  return number;
}

#endif
