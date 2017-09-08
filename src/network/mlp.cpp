#ifdef _MLP_HPP_

template <typename T>
MLP<T>::MLP()
  : full1{N_UNITS, N_IN}
  , full2{N_UNITS, N_UNITS}
  , full3{N_OUT, N_UNITS}
  , bn1(16)
  , bn2(N_UNITS)
  , bn3(N_OUT)
{
  input       = zeros<T>(1, IMHEI, IMWID);
  input_flat  = zeros<T>(1 * IMHEI * IMWID);
  unit1       = zeros<T>(N_UNITS);
  aunit1      = zeros<T>(N_UNITS);
  bunit1      = zeros<T>(N_UNITS);
  unit2       = zeros<T>(N_UNITS);
  aunit2      = zeros<T>(N_UNITS);
  bunit2      = zeros<T>(N_UNITS);
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
  bn1.load(path+"/bn_1");
  full2.load(path+"/wb_2");
  bn2.load(path+"/bn_2");
  full3.load(path+"/wb_3");
  bn3.load(path+"/bn_3");
}

template <typename T>
void MLP<T>::Save(string path)
{
}

template <typename T>
void MLP<T>::Forward(string data)
{
}

template <typename T>
void MLP<T>::Backward(int label)
{
}

template <typename T>
void MLP<T>::Update()
{
}
template <typename T>
int MLP<T>::calc(string data)
{
  load_image(data, input);

  flatten(input, input_flat);

  full1.forward(input_flat, unit1);
    // bn1.forward(unit1, bunit1);
  relu1.forward(unit1, aunit1);
  full2.forward(aunit1, unit2);
    // bn2.forward(unit2, bunit2);
  relu2.forward(unit2, aunit2);
  full3.forward(aunit2, output);
    // bn3.forward(unit3, output);

  int number = -1;
  int temp = INT_MIN;
  for (int i=0; i<N_OUT; i++) {
    if (temp < output[i]) {
       temp = output[i];
       number = i;
    }
  }

  return number;
}

#endif
