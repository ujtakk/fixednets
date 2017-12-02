#ifdef _CONVMODULE_HPP_

template <typename T>
ConvModule<T>::ConvModule(int n_out, int n_in, int fil_h, int fil_w,
                          int stride, int pad)
  : conv{n_out, n_in, fil_h, fil_w, stride, pad}
  , shape{n_out, n_in, fil_h, fil_w}
  , stride(stride), pad(pad)
{
}

template <typename T>
ConvModule<T>::~ConvModule()
{
}

template <typename T>
void ConvModule<T>::load(std::string path)
{
  conv.load(path);
}

template <typename T>
void ConvModule<T>::save(std::string path)
{
}

template <typename T>
void ConvModule<T>::forward(Mat3D<T>& output, Mat3D<T>& input)
{
  const int n_out = shape[0];
  const int out_h = (input[0].size()    + 2*pad - shape[2] + 1) / stride;
  const int out_w = (input[0][0].size() + 2*pad - shape[3] + 1) / stride;

  // Mat3D<T> conved = zeros<T>(n_out, out_h, out_w);
  Mat3D<T> conved;
  output = zeros<T>(n_out, out_h, out_w);

  conv.forward(conved, input);
  relu.forward(output, conved);
}

template <typename T>
void ConvModule<T>::backward(Mat3D<T>& input, Mat3D<T>& output)
{
}

template <typename T>
void ConvModule<T>::update()
{
}

#endif
