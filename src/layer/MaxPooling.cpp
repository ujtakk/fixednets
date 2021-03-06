#ifdef _MAXPOOLING_HPP_

#include <limits>

template <typename T>
MaxPooling<T>::MaxPooling(const int pool_h, const int pool_w, int stride, int pad)
  : shape{pool_h, pool_w}
  , stride(stride), pad(pad)
{
}

template <typename T>
MaxPooling<T>::~MaxPooling()
{
}

template <typename T>
void MaxPooling<T>::forward(Mat3D<T>& output, Mat3D<T>& input)
{
  const int n_out = input.size();
  // const int out_h = (input[0].size()    + 2*pad - shape[0] + stride) / stride;
  // const int out_w = (input[0][0].size() + 2*pad - shape[1] + stride) / stride;
  const int out_h = (input[0].size()    + 2*pad - shape[0] + 1) / stride;
  const int out_w = (input[0][0].size() + 2*pad - shape[1] + 1) / stride;
  output = zeros<T>(n_out, out_h, out_w);

  pool_max(output, input, shape[0], shape[1], stride, pad);
}

//propagate grads only for position of max in filter
template <typename T>
void MaxPooling<T>::backward(Mat3D<T>& input, Mat3D<T>& output)
{
  const int pmnum = output.size();
  const int pmhei = output[0].size();
  const int pmwid = output[0][0].size();

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < pmnum; n++) {
    for (int i = 0; i < pmhei; i++) {
      for (int j = 0; j < pmwid; j++) {
        int max = std::numeric_limits<T>::min();

        for (int k = 0; k < shape[0]; k++) {
          for (int l = 0; l < shape[1]; l++) {
            if (input[n][i*shape[0]+k][j*shape[1]+l] > max) {
              max = input[n][i*shape[0]+k][j*shape[1]+l];
              input[n][i*shape[0]+k][j*shape[1]+l] = output[n][i][j];
            }
            else {
              input[n][i*shape[0]+k][j*shape[1]+l] = 0;
            }
          }
        }
      }
    }
  }
}

#endif
