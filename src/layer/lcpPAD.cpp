#ifdef _LCPPAD_HPP_

using std::to_string;

#define THRES 256

template <typename T>
lcpPAD<T>::lcpPAD(int out_channels, int in_channels, const int f_height, const int f_width, const int phei, const int pwid, int cstride, int cpad, int pstride)
  : cshape{out_channels, in_channels, f_height, f_width}
  , pshape{phei, pwid}
{
  iw = zeros<T>(out_channels, in_channels, f_height, f_width);
  gw = zeros<T>(out_channels, in_channels, f_height, f_width);
  ib = zeros<T>(out_channels);
  gb = zeros<T>(out_channels);
  this->cstride = cstride;
  this->cpad    = cpad;
  this->pstride = pstride;
}

template <typename T>
lcpPAD<T>::~lcpPAD()
{
}

template <typename T>
void lcpPAD<T>::load(string path)
{
  vector<string> filename(cshape[0]);

  if (cshape[1] == 1) {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i<cshape[0]; i++) {
      filename[i] = path+"/data"+to_string(i)+".txt";
      load_data(filename[i], iw[i][0], ib[i], cshape[2], cshape[3]);
    }
  }
  else {
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0; i<cshape[0]; i++) {
      for (int j=0; j<cshape[1]; j++) {
        filename[i] = path+"/data"+to_string(i)+"_"+to_string(j)+".txt";
        load_w(filename[i], iw[i][j], cshape[2], cshape[3]);
      }

      filename[i] = path+"/data"+to_string(i)+".txt";
      load_b(filename[i], ib[i]);
    }
  }
}

// TODO: Make if padded ver.
template <typename T>
void lcpPAD<T>::forward(Mat3D<T> &input, Mat3D<T> &output, int which, int amount)
{
  const int ihei = input[0].size();
  const int iwid = input[0][0].size();
  const int fmhei = (ihei-cshape[2]+cstride+2*cpad)/cstride;
  const int fmwid = (iwid-cshape[3]+cstride+2*cpad)/cstride;
  const int pmhei = (fmhei-pshape[0]+pstride) / pstride;
  const int pmwid = (fmwid-pshape[1]+pstride) / pstride;

  //int max = std::numeric_limits<T>::min();
  //int x, y;
  //int p, s;

  Mat1D<T> max(cshape[0], std::numeric_limits<T>::min());
  Mat1D<T> x(cshape[0], 0);
  Mat1D<T> y(cshape[0], 0);
  Mat1D<T> p(cshape[0], 0);
  Mat1D<T> s(cshape[0], 0);

  Mat3D<T> sum;
  Mat3D<T> pro;
  Mat3D<T> total;

  Mat3D<T> padded;
  Mat3D<T> in_trunc;
//  Mat3D<T> out_trunc;

  Mat4D<T> w_trunc;
  Mat4D<T> index;

  std::random_device rd;
  std::mt19937 mt(rd());

  sum = zeros<T>(cshape[0], fmhei, fmwid);
  pro = zeros<T>(cshape[0], fmhei, fmwid);
  total = zeros<T>(cshape[0], pmhei, pmwid);

  //padded = zeros<T>(cshape[1], ihei+cshape[2]-1, iwid+cshape[3]-1);
  padded = zeros<T>(cshape[1], ihei+2*cpad, iwid+2*cpad);
  //in_trunc = zeros<T>(cshape[1], ihei+cshape[2]-1, iwid+cshape[3]-1);
  in_trunc = zeros<T>(cshape[1], ihei+2*cpad, iwid+2*cpad);
//  out_trunc = zeros<T>(cshape[0], fmhei, fmwid);

  w_trunc = zeros<T>(cshape[0], cshape[1], cshape[2], cshape[3]);
  index = zeros<T>(cshape[0], pmhei, pmwid, 2);

  //printf("truncation\n");

/*  ----------- use as padded -----------*/
  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int m=0; m<cshape[1]; m++) {
    for (int i=0; i<ihei; i++) {
      for (int j=0; j<iwid; j++) {
           padded[m][i+cpad][j+cpad] = input[m][i][j];
         in_trunc[m][i+cpad][j+cpad] = input[m][i][j];// >> amount;
        //in_trunc[m][i][j] = in_trunc[m][i][j] << amount;
      }
    }
  }

  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int i=0; i<cshape[0]; i++) {
    for (int j=0; j<cshape[1]; j++) {
      for (int k=0; k<cshape[2]; k++) {
        for (int l=0; l<cshape[3]; l++) {
          w_trunc[i][j][k][l] = iw[i][j][k][l];// >> amount;
          //w_trunc[i][j][k][l] = w_trunc[i][j][k][l] << amount;
        }
      }
    }
  }

  //printf("first convolution\n");
  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n=0; n<cshape[0]; n++) {
    for (int m=0; m<cshape[1]; m++) {
      for (int i=0; i<ihei+2*cpad-cshape[2]+1; i+=cstride) {
        for (int j=0; j<iwid+2*cpad-cshape[3]+1; j+=cstride) {
          for (int k=0; k<cshape[2]; k++) {
            for (int l=0; l<cshape[3]; l++) {
              /* No Approximation */
              //if (in_trunc[m][i+k][j+l] * w_trunc[n][m][k][l] >= 0)
              //  p[n] = (in_trunc[m][i+k][j+l] * w_trunc[n][m][k][l]) / Q_OFFSET<T>;
              //else
              //  p[n] = ((in_trunc[m][i+k][j+l] * w_trunc[n][m][k][l]) / Q_OFFSET<T>) - 1;
              /* Sign Connect */
              //if (w_trunc[n][m][k][l] >= 0) p[n] = in_trunc[m][i+k][j+l];
              //else if (w_trunc[n][m][k][l] < 0) p[n] = -in_trunc[m][i+k][j+l];
              /* Ternary Connect */
              if ( ((int)mt() % THRES < abs(w_trunc[n][m][k][l])) && (w_trunc[n][m][k][l] > 0) ) p[n] = in_trunc[m][i+k][j+l];
              else if ( ((int)mt() % THRES < abs(w_trunc[n][m][k][l])) && (w_trunc[n][m][k][l] < 0) ) p[n] = -in_trunc[m][i+k][j+l];
              else p[n] = 0;
              s[n] += p[n];
              //s[n] = ADD(16, s[n], p[n], which, amount);
            }
          }
          sum[n][i][j] += s[n];
          s[n] = 0;
        }
      }
    }

    // for (int k=0; k<fmhei; k++) {
    //   for (int l=0; <fmwid; l++) {
    //     out_trunc[n][k][l] = sum[n][k][l];
    //     sum[n][k][l] = 0;
    //   }
    // }
  }

  //printf("max pooling\n");
  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n=0; n<cshape[0]; n++) {
    for (int i=0; i<fmhei-pshape[0]+pstride; i+=pstride) {
      for (int j=0; j<fmwid-pshape[1]+pstride; j+=pstride) {
        for (int k=0; k<pshape[0]; k++) {
          for (int l=0; l<pshape[1]; l++) {
            if (sum[n][i+k][j+l] > max[n]) {
              max[n] = sum[n][i+k][j+l];
              index[n][i/pstride][j/pstride][0] = i+k;
              index[n][i/pstride][j/pstride][1] = j+l;
            }
          }
        }
        max[n]=std::numeric_limits<T>::min();
      }
    }
  }

  //printf("second convolution\n");
  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n=0; n<cshape[0]; n++) {
    for (int m=0; m<cshape[1]; m++) {
      for (int i=0; i<pmhei; i++) {
        for (int j=0; j<pmwid; j++) {
          x[n] = index[n][i][j][0];
          y[n] = index[n][i][j][1];

          for (int k=0; k<cshape[2]; k++) {
            for (int l=0; l<cshape[3]; l++) {
              if (padded[m][x[n]+k][y[n]+l] * iw[n][m][k][l] >= 0)
                p[n] = (padded[m][x[n]+k][y[n]+l] * iw[n][m][k][l]) / Q_OFFSET<T>;
              else
                p[n] = ((padded[m][x[n]+k][y[n]+l] * iw[n][m][k][l]) / Q_OFFSET<T>) - 1;
              s[n] += p[n];
            }
          }
          total[n][i][j] += s[n];
          s[n] = 0;
        }
      }
    }
    for (int i=0; i<pmhei; i++) {
      for (int j=0; j<pmwid; j++) {
        output[n][i][j] = total[n][i][j] + ib[n];
        total[n][i][j] = 0;
      }
    }
  }
}

#endif
