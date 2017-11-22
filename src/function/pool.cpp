#ifdef _POOL_HPP_

#include <limits>

/*median pooling*/
static void swap(Mat1D<int>& input,int i,int j)
{
  int temp;

  temp = input[i];
  input[i] = input[j];
  input[j] = temp;
}

void median_pooling(
  Mat2D<int>& fmap, Mat2D<int>& pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
)
{
  Mat1D<int> cluster(phei*pwid);
  for (int i = 0; i < fmhei; i = i+phei) {
    for (int j = 0; j < fmwid; j = j+pwid) {
      for (int k = 0; k < phei; k++) {
        for (int l = 0; l < pwid; l++) {
          cluster[k*pwid+l] = fmap[i+k][j+l];
        }
      }
      for (int k = 0; k < phei*pwid-1; k++) {
        for (int l = phei*pwid-1; l>k; l--) {
          if (cluster[l-1]>cluster[l])
            swap(cluster,l-1,l);
        }
      }

      pmap[i/phei][j/pwid] = (cluster[1]+cluster[2])/2;
    }
  }
}

template <typename T>
void pool_max(Mat3D<T>& input, Mat3D<T>& output,
              int fil_h, int fil_w, int stride)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  const int fea_h = in_h - fil_h + stride;
  const int fea_w = in_w - fil_w + stride;

  Mat1D<T> max(n_in, std::numeric_limits<T>::min());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_in; ++n) {
    for (int i = 0; i < fea_h; i+=stride) {
      for (int j = 0; j < fea_w; j+=stride) {
        for (int k = 0; k < fil_h; ++k) {
          for (int l = 0; l < fil_w; ++l) {
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

template <typename T>
void pool_max(Mat3D<T>& input, int fil_h, int fil_w, int stride)
{
  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();
  const int fea_h = in_h - fil_h + stride;
  const int fea_w = in_w - fil_w + stride;

  auto output = zeros<T>(n_in, fea_h/stride, fea_w/stride);

  Mat1D<T> max(n_in, std::numeric_limits<T>::min());

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < n_in; ++n) {
    for (int i = 0; i < fea_h; i+=stride) {
      for (int j = 0; j < fea_w; j+=stride) {
        for (int k = 0; k < fil_h; ++k) {
          for (int l = 0; l < fil_w; ++l) {
            if (input[n][i+k][j+l] > max[n])
              max[n] = input[n][i+k][j+l];
          }
        }
        output[n][i/stride][j/stride] = max[n];
        max[n] = std::numeric_limits<T>::min();
      }
    }
  }

  return output;
}

#endif
