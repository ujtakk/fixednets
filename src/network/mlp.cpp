#ifdef _MLP_HPP_

#include <limits>

template <typename T>
MLP<T>::MLP()
  : full1{N_UNITS, N_IN}
  , full2{N_OUT, N_UNITS}
{
  // input       = zeros<T>(1, IMHEI, IMWID);
  input_flat  = zeros<T>(1 * IMHEI * IMWID);
  // unit1       = zeros<T>(N_UNITS);
  // aunit1      = zeros<T>(N_UNITS);
  // unit2       = zeros<T>(N_OUT);
  // output      = zeros<T>(N_OUT);
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
  flatten(input_flat, input);

  full1.forward(unit1, input_flat);
  relu1.forward(aunit1, unit1);
  full2.forward(unit2, aunit1);

  prob2.prob(output, unit2);
}

#include <cstdio>
template <typename T>
void MLP<T>::Backward(int label)
{
  prob2.loss(unit2, output, label);

  full2.backward(aunit1, unit2);
  relu1.backward(unit1, aunit1);
  full1.backward(input_flat, unit1);
}

template <typename T>
void MLP<T>::Update()
{
  full1.update();
  full2.update();
}

template <typename T>
int MLP<T>::calc(std::string data)
{
  // load_txt(input, data);
  // flatten(input_flat, input);
  load_txt(input_flat, data);

  full1.forward(unit1, input_flat);
  relu1.forward(aunit1, unit1);
  full2.forward(output, aunit1);
  // full2.forward(unit2, aunit1);
  // prob2.forward(output, unit2);

  return classify(output);
}

#endif
