#ifdef _MEDIANPOOLING_HPP_

template <typename T>
MedianPooling<T>::MedianPooling(const int phei, const int pwid)
  :   shape{phei, pwid}
{
}

template <typename T>
MedianPooling<T>::~MedianPooling()
{
}

template <typename T>
void MedianPooling<T>::forward(
  Mat3D<T> &input,
  Mat3D<T> &output
  )
{
  const int fmnum = input.size();
  const int fmhei = input[0].size();
  const int fmwid = input[0][0].size();

  Mat1D<T> cluster(shape[0]*shape[1]);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < fmnum; n++) {
    for (int i = 0; i < fmhei; i = i+shape[0]) {
      for (int j = 0; j < fmwid; j = j+shape[1]) {
        for (int k = 0; k < shape[0]; k++) {
          for (int l = 0; l < shape[1]; l++) {
            cluster[k*shape[1]+l] = input[n][i+k][j+l];
          }
        }

        for (int k = 0; k < shape[0]*shape[1]-1; k++) {
          for (int l = shape[0]*shape[1]-1; l>k; l--) {
            if (cluster[l-1] > cluster[l]) {
              swap(cluster,l-1,l);
            }
          }
        }

        if (shape[0]*shape[1] % 2 == 0) {
          output[n][i/shape[0]][j/shape[1]]
            = ( cluster[(shape[0]*shape[1])/2-1] + cluster[(shape[0]*shape[1])/2] ) / 2;
        }
        else {
          output[n][i/shape[0]][j/shape[1]]
            = cluster[ (shape[0]*shape[1]-1)/2 ];
        }
      }
    }
  }
}

template <typename T>
void MedianPooling<T>::backward(
  Mat3D<T> &output,
  Mat3D<T> &input
  )
{
  const int pmnum = output.size();
  const int pmhei = output[0].size();
  const int pmwid = output[0][0].size();

  Mat1D<T> cluster(shape[0]*shape[1]);
  Mat1D<T> arg(shape[0]*shape[1]);

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int n = 0; n < pmnum; n++) {
    for (int i = 0; i < pmhei; i++) {
      for (int j = 0; j < pmwid; j++) {
        for (int k = 0; k < shape[0]; k++) {
          for (int l = 0; l < shape[1]; l++) {
            cluster[k*shape[1]+l] = input[n][i*shape[0]+k][j*shape[1]+l];
            arg[k*shape[1]+l] = k*shape[1]+l;
          }
        }

        for (int k = 0; k < shape[0]*shape[1]-1; k++) {
          for (int l = shape[0]*shape[1]-1; l>k; l--) {
            if (cluster[l-1] > cluster[l]) {
              swap(cluster, l-1, l);
              swap(arg, l-1, l);
            }
          }
        }

        for (int k = 0; k < shape[0]*shape[1]; k++) {
          if (
            (shape[0]*shape[1]) % 2 == 0
            && (k == (shape[0]*shape[1])/2-1 || k == (shape[0]*shape[1])/2)
          ) {
            input[n][ i*shape[0] + arg[k]/shape[1] ][ j*shape[1] + arg[k]%shape[1] ]
              = output[n][i][j] / 2;
          }
          else if ((shape[0]*shape[1]) % 2 == 1 && k == (shape[0]*shape[1]-1)/2) {
            input[n][ i*shape[0] + arg[k]/shape[1] ][ j*shape[1] + arg[k]%shape[1] ]
              = output[n][i][j];
          }
          else {
            input[n][ i*shape[0] + arg[k]/shape[1] ][ j*shape[1] + arg[k]%shape[1] ]
              = 0;
          }
        }
      }
    }
  }
}

#endif
