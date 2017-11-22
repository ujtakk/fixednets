#ifdef _CONVMODULE_HPP_

template <typename T>
ConvModule<T>::ConvModule(int n_out, int n_in, int fil_h, int fil_w,
                          int stride, int pad)
  : conv{n_out, n_in, fil_h, fil_w, stride, pad}
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
void ConvModule<T>::forward(Mat3D<T>& input, Mat3D<T>& output)
{
  const int n_out = output.size();
  const int out_h = output[0].size();
  const int out_w = output[0][0].size();

  Mat3D<T> conved = zeros<T>(n_out, out_h, out_w);

  conv.forward(input, conved);
  relu.forward(conved, output);
}

template <typename T>
void ConvModule<T>::backward(Mat3D<T>& output, Mat3D<T>& input)
{
}

template <typename T>
void ConvModule<T>::update()
{
}

#endif
