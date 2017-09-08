#ifdef _MAXPOOLING_HPP_

#include <limits>

template <typename T>
MaxPooling<T>::MaxPooling(const int phei, const int pwid, int stride)
  :   shape{phei, pwid}
{
  this->stride = stride;
}

template <typename T>
MaxPooling<T>::~MaxPooling()
{
}

template <typename T>
void MaxPooling<T>::forward(Mat3D<T> &input, Mat3D<T> &output)
{
  const int fmnum = input.size();
  const int fmhei = input[0].size();
  const int fmwid = input[0][0].size();

  Mat1D<T> max(fmnum, std::numeric_limits<T>::min());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < fmnum; n++) {
    for (int i = 0; i < fmhei-shape[0]+stride; i+=stride) {
      for (int j = 0; j < fmwid-shape[1]+stride; j+=stride) {
        for (int k = 0; k < shape[0]; k++) {
          for (int l = 0; l < shape[1]; l++) {
            if (input[n][i+k][j+l] > max[n])
              max[n] = input[n][i+k][j+l];
          }
        }
        output[n][i/stride][j/stride] = max[n];
        max[n] = std::numeric_limits<T>::min();
      }
    }
  }
}

//propagate grads only for position of max in filter
template <typename T>
void MaxPooling<T>::backward(Mat3D<T> &output, Mat3D<T> &input)
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
