#ifdef _FIREMODULE_HPP_

template <typename T>
FireModule<T>::FireModule(int s1x1, int e1x1, int e3x3, int n_in)
  : squeeze1x1{s1x1, n_in, 1, 1, 1, 0}
  , expand1x1{e1x1, s1x1, 1, 1, 1, 0}
  , expand3x3{e3x3, s1x1, 3, 3, 1, 1}
  , s1x1(s1x1) , e1x1(e1x1) , e3x3(e3x3) , n_in(n_in)
{
}

template <typename T>
FireModule<T>::~FireModule()
{
}

template <typename T>
void FireModule<T>::load(std::string path)
{
  squeeze1x1.load(path+"/squeeze1x1");
  expand1x1.load(path+"/expand1x1");
  expand3x3.load(path+"/expand3x3");
}

template <typename T>
void FireModule<T>::save(std::string path)
{
}

template <typename T>
void FireModule<T>::forward(Mat3D<T>& output, Mat3D<T>& input)
{
  const int out_h = output[0].size();
  const int out_w = output[0][0].size();

  Mat3D<T> sq1x1 = zeros<T>(s1x1, out_h, out_w);
  Mat3D<T> ex1x1 = zeros<T>(e1x1, out_h, out_w);
  Mat3D<T> ex3x3 = zeros<T>(e3x3, out_h, out_w);

  squeeze1x1.forward(sq1x1, input);
  expand1x1.forward(ex1x1, sq1x1);
  expand3x3.forward(ex3x3, sq1x1);

  concat(output, ex1x1, ex3x3);
}

template <typename T>
void FireModule<T>::backward(Mat3D<T>& input, Mat3D<T>& output)
{
}

template <typename T>
void FireModule<T>::update()
{
}

#endif
