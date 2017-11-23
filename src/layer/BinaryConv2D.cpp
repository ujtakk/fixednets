#ifdef _BINARYCONV2D_HPP_

template <typename T>
BinaryConv2D<T>::BinaryConv2D(int out_channels, int in_channels,
  const int f_height, const int f_width, int stride, int pad)
  : shape{out_channels, in_channels, f_height, f_width}
{
  iw = zeros<T>(out_channels, in_channels, f_height, f_width);
  gw = zeros<T>(out_channels, in_channels, f_height, f_width);
  ib = zeros<T>(out_channels);
  gb = zeros<T>(out_channels);
  this->stride = stride;
  this->pad  = pad;
}

template <typename T>
BinaryConv2D<T>::~BinaryConv2D()
{
}

template <typename T>
void BinaryConv2D<T>::load(std::string path)
{
  load_txt(iw, path+"/W.txt");
  load_txt(ib, path+"/b.txt");
}

template <typename T>
void BinaryConv2D<T>::save(std::string path)
{
}

template <typename T>
void BinaryConv2D<T>::forward(Mat3D<T>& output, Mat3D<T>& input)
{
  const int ihei = input[0].size();
  const int iwid = input[0][0].size();

  Mat3D<T> sum;
  Mat3D<T> pro;

  pro = zeros<T>(shape[0], (ihei-shape[2]+stride+2*pad)/stride, (iwid-shape[3]+stride+2*pad)/stride);
  sum = zeros<T>(shape[0], (ihei-shape[2]+stride+2*pad)/stride, (iwid-shape[3]+stride+2*pad)/stride);

  // for (int i = 0; i < shape[0]; i++) {
  //  for (int j = 0; j < shape[1]; j++) {
  //    conv_plus(input[j], iw[i][j], pro[i], ihei, iwid, shape[2], shape[3]);
  //    for (int k = 0; k < ihei-shape[2]+stride; k++) {
  //      for (int l = 0; l < iwid-shape[3]+stride; l++) {
  //        sum[i][k][l] += pro[i][k][l]; } } }

   // for (int k = 0; k < ihei-shape[2]+stride; k++) {
   //   for (int l = 0; l < iwid-shape[3]+stride; l++) {
   //     output[i][k][l] = sum[i][k][l] + ib[i];
   //     sum[i][k][l] = 0; } } }

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int i = 0; i < shape[0]; i++) {
    for (int j = 0; j < shape[1]; j++) {
      conv_plus_bi(input[j], iw[i][j], pro[i], ihei, iwid, shape[2], shape[3], stride, pad);

      for (int k = 0; k < (ihei+2*pad-shape[2]+stride)/stride; k++) {
        for (int l = 0; l < (iwid+2*pad-shape[3]+stride)/stride; l++) {
          sum[i][k][l] += pro[i][k][l];
        }
      }
    }

    for (int k = 0; k < (ihei-shape[2]+stride+2*pad)/stride; k++) {
      for (int l = 0; l < (iwid-shape[3]+stride+2*pad)/stride; l++) {
        output[i][k][l] = sum[i][k][l] + ib[i];
        sum[i][k][l] = 0;
      }
    }
  }
}

#endif
