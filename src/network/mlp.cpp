#ifdef _MLP_HPP_

#include <limits>

template <typename T>
MLP<T>::MLP()
  : full1{N_UNITS, N_IN}
  , full2{N_UNITS, N_UNITS}
  , full3{N_OUT, N_UNITS}
{
  input       = zeros<T>(1, IMHEI, IMWID);
  input_flat  = zeros<T>(1 * IMHEI * IMWID);
  unit1       = zeros<T>(N_UNITS);
  aunit1      = zeros<T>(N_UNITS);
  unit2       = zeros<T>(N_UNITS);
  aunit2      = zeros<T>(N_UNITS);
  unit3       = zeros<T>(N_OUT);
  output      = zeros<T>(N_OUT);
}

template <typename T>
MLP<T>::~MLP()
{
}

template <typename T>
void MLP<T>::Load(string path)
{
  full1.load(path+"/wb_1");
  full2.load(path+"/wb_2");
  full3.load(path+"/wb_3");
}

template <typename T>
void MLP<T>::Save(string path)
{
  full1.save(path+"/wb_1");
  full2.save(path+"/wb_2");
  full3.save(path+"/wb_3");
}

template <typename T>
void MLP<T>::Forward(string data)
{
  load_image(data, input);

  full1.forward(input_flat, unit1);
  relu1.forward(unit1, aunit1);
  full2.forward(aunit1, unit2);
  relu2.forward(unit2, aunit2);
  full3.forward(aunit2, unit3);

  prob3.forward(unit3, output);
}

template <typename T>
void MLP<T>::Backward(int label)
{
  prob3.loss(label, output, unit3);

  full3.backward(unit3, aunit2);
  relu2.backward(aunit2, unit2);
  full2.backward(unit2, aunit1);
  relu1.backward(aunit1, unit1);
  full1.backward(unit1, input_flat);
}

template <typename T>
void MLP<T>::Update()
{
  full1.update();
  full2.update();
  full3.update();
}
template <typename T>
int MLP<T>::calc(string data)
{
  load_image(data, input);

  flatten(input, input_flat);

  full1.forward(input_flat, unit1);
  relu1.forward(unit1, aunit1);
  full2.forward(aunit1, unit2);
  relu2.forward(unit2, aunit2);
  full3.forward(aunit2, output);

  int number = -1;
  int temp = std::numeric_limits<int>::min();
  for (int i=0; i<N_OUT; i++) {
    if (temp < output[i]) {
       temp = output[i];
       number = i;
    }
  }

  return number;
}

#endif
