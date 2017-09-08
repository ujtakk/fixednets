#ifdef _LAZYCONVPOOL_HPP_

#include <random>

using std::to_string;

template <typename T>
LazyConvPool<T>::LazyConvPool(int out_channels, int in_channels, const int f_height, const int f_width, const int phei, const int pwid)
  : cshape{out_channels, in_channels, f_height, f_width}
  , pshape{phei, pwid}
{
  iw = zeros<T>(out_channels, in_channels, f_height, f_width);
  gw = zeros<T>(out_channels, in_channels, f_height, f_width);
  ib = zeros<T>(out_channels);
  gb = zeros<T>(out_channels);
}

template <typename T>
LazyConvPool<T>::~LazyConvPool()
{
}

template <typename T>
void LazyConvPool<T>::load(string path)
{
  string filename;

  if (cshape[1]==1) {
    for (int i=0; i<cshape[0]; i++) {
      filename = path+"/data"+to_string(i)+".txt";
      load_data(filename, iw[i][0], ib[i], cshape[2], cshape[3]);
    }
  }
  else {
    for (int i=0; i<cshape[0]; i++) {
      for (int j=0; j<cshape[1]; j++) {
        filename = path+"/data"+to_string(i)+"_"+to_string(j)+".txt";
        load_w(filename, iw[i][j], cshape[2], cshape[3]);
      }
    }

    for (int i=0; i<cshape[1]; i++) {
      filename = path+"/data"+to_string(i)+".txt";
      load_b(filename, ib[i]);
    }
  }
}

/*
TODO: truncate operands of first convolution;
    have array containing positions which indicate the value chosen at fmap;
    calculate second convolution only for areas indicated by the array;
*/
template <typename T>
void LazyConvPool<T>::forward(Mat3D<T> &input, Mat3D<T> &output, int which, int amount)
{

// lazy() is implemented in func.cpp
/*----------  Implementation 1  ----------*/

//  lazy(input, iw, ib, output, cshape[0], cshape[1], cshape[2], cshape[3], pshape[0], pshape[1]);

// appconv(), maxindex(), preconv() is implemented in func.cpp
/*----------  Implementation 2  ----------*/

//  const int ihei = input[0].size();
//  const int iwid = input[0][0].size();
//  const int fmhei = ihei-cshape[2]+1;
//  const int fmwid = iwid-cshape[3]+1;
//  const int pmhei = fmhei / pshape[0];
//  const int pmwid = fmwid / pshape[1];
//  vector<vector<vector<int>>> out_trunc;
//  vector<vector<vector<vector<int>>>> index;
//  out_trunc = zeros<int>(cshape[0], fmhei, fmwid);
//  index = zeros<int>(cshape[0], pmhei, pmwid, 2);
//  appconv(input, iw, out_trunc, ihei, iwid, cshape[0], cshape[1], cshape[2], cshape[3]);
//  maxindex(out_trunc, index, ihei, iwid, cshape[0], cshape[2], cshape[3], pshape[0], pshape[1]);
//  preconv(index, input, iw, ib, output, ihei, iwid, cshape[0], cshape[1], cshape[2], cshape[3], pshape[0], pshape[1]);

/*----------  Implementation 3  ----------*/

  const int ihei = input[0].size();
  const int iwid = input[0][0].size();
  const int fmhei = ihei-cshape[2]+1;
  const int fmwid = iwid-cshape[3]+1;
  const int pmhei = fmhei / pshape[0];
  const int pmwid = fmwid / pshape[1];

  Mat1D<T> max(cshape[0], std::numeric_limits<T>::min());
  Mat1D<T> x(cshape[0], 0);
  Mat1D<T> y(cshape[0], 0);
  Mat1D<T> p(cshape[0], 0);
  Mat1D<T> s(cshape[0], 0);
  Mat3D<T> sum;
  Mat3D<T> pro;
  Mat3D<T> total;

  //Mat3D<T> in_trunc;
  //Mat3D<T> w_eval;
  Mat3D<T> out_trunc;
  Mat3D<T> out_eval;

  //Mat4D<T> w_trunc;
  //Mat4D<T> in_eval;
  Mat4D<T> index;
  Mat4D<T> index_ref;

  std::random_device rd;
  std::mt19937 mt(rd());

  sum = zeros<T>(cshape[0], fmhei, fmwid);
  pro = zeros<T>(cshape[0], fmhei, fmwid);
  total = zeros<T>(cshape[0], pmhei, pmwid);

  //in_trunc = zeros<T>(cshape[1], ihei, iwid);
  //in_eval = zeros<T>(cshape[1], fmhei, fmwid, cshape[2]*cshape[3]);
  out_trunc = zeros<T>(cshape[0], fmhei, fmwid);
  out_eval = zeros<T>(cshape[0], fmhei, fmwid);

  //w_trunc = zeros<T>(cshape[0], cshape[1], cshape[2], cshape[3]);
  //w_eval = zeros<T>(cshape[0], cshape[1], cshape[2]*cshape[3]);
  index = zeros<T>(cshape[0], pmhei, pmwid, 2);
  index_ref = zeros<T>(cshape[0], pmhei, pmwid, 2);


  /*  ----------- use as truncated -----------*/
  // for (int m=0; m<cshape[1]; m++) {
  //   for (int i=0; i<ihei; i++) {
  //     for (int j=0; j<iwid; j++) {
  //       in_trunc[m][i][j] = input[m][i][j];// >> amount;
  //       //in_trunc[m][i][j] = in_trunc[m][i][j] << amount;
  //     }
  //   }
  // }
  //
  // for (int i=0; i<cshape[0]; i++) {
  //   for (int j=0; j<cshape[1]; j++) {
  //     for (int k=0; k<cshape[2]; k++) {
  //       for (int l=0; l<cshape[3]; l++) {
  //         w_trunc[i][j][k][l] = iw[i][j][k][l];// >> amount;
  //         //w_trunc[i][j][k][l] = w_trunc[i][j][k][l] << amount;
  //       }
  //     }
  //   }
  // }

//  printf("first convolution\n");
  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n=0; n<cshape[0]; n++) {
    for (int m=0; m<cshape[1]; m++) {
      for (int i=cshape[2]-1; i<ihei; i++) {
        for (int j=cshape[3]-1; j<iwid; j++) {
          for (int k=0; k<cshape[2]; k++) {
            for (int l=0; l<cshape[3]; l++) {
              p[n] = (input[m][i-k][j-l] * iw[n][m][k][l]) / Q_OFFSET<T>;

              //if (iw[n][m][k][l] >= 0) p[n] = input[m][i-k][j-l];
              //else if (iw[n][m][k][l] < 0) p[n] = -input[m][i-k][j-l];

              //if ( ((int)mt() % THRES < abs(iw[n][m][k][l])) && (iw[n][m][k][l] > 0)) p[n] = input[m][i-k][j-l];
              //else if ( ((int)mt() % THRES < abs(iw[n][m][k][l])) && (iw[n][m][k][l] < 0) ) p[n] = -input[m][i-k][j-l];
              //else p[n] = 0;

              s[n] += p[n];
              //s[n] = ADD(16, s[n], p[n], which, amount);
            }
          }
          sum[n][i-cshape[2]+1][j-cshape[3]+1] += s[n];
          s[n] = 0;
        }
      }
    }

    // for (int k=0; k<fmhei; k++) {
    //   for (int l=0; l<fmwid; l++) {
    //     out_trunc[n][k][l] = sum[n][k][l];
    //     sum[n][k][l] = 0;
    //   }
    // }
  }

//  printf("max pooling\n");
  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n=0; n<cshape[0]; n++) {
    for (int i=0; i<pmhei; i++) {
      for (int j=0; j<pmwid; j++) {
        for (int k=0; k<pshape[0]; k++) {
          for (int l=0; l<pshape[1]; l++) {
            if (sum[n][i*pshape[0]+k][j*pshape[1]+l] > max[n])
            {
              max[n] = sum[n][i*pshape[0]+k][j*pshape[1]+l];
              index[n][i][j][0] = i*pshape[0] + k;
              index[n][i][j][1] = j*pshape[1] + l;
            }
            sum[n][i*pshape[0]+k][j*pshape[1]+l] = 0;
          }
        }
        max[n]=std::numeric_limits<T>::min();
      }
    }
  }

//  printf("second convolution\n");
  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n=0; n<cshape[0]; n++) {
    for (int m=0; m<cshape[1]; m++) {
      for (int i=0; i<pmhei; i++) {
        for (int j=0; j<pmwid; j++) {
          x[n] = index[n][i][j][0] + cshape[2] - 1;
          y[n] = index[n][i][j][1] + cshape[3] - 1;

          for (int k=0; k<cshape[2]; k++) {
            for (int l=0; l<cshape[3]; l++) {
              p[n] = (input[m][x[n]-k][y[n]-l] * iw[n][m][k][l]) >> 8;
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

  // #ifdef _openmp
  // #pragma omp parallel for
  // #endif
  // for (int n=0; n<cshape[0]; n++) {
  //   for (int m=0; m<cshape[1]; m++) {
  //     conv(input[m], iw[n][m], pro[n], ihei, iwid, cshape[2], cshape[3]);
  //
  //     for (int k=0; k<fmhei; k++) {
  //       for (int l=0; l<fmwid; l++) {
  //         sum[n][k][l] += pro[n][k][l];
  //       }
  //     }
  //   }
  //   for (int k=0; k<fmhei; k++) {
  //     for (int l=0; l<fmwid; l++) {
  //       out_eval[n][k][l] = sum[n][k][l] + ib[n];
  //       sum[n][k][l] = 0;
  //     }
  //   }
  // }
  //
  // #ifdef _openmp
  // #pragma omp parallel for
  // #endif
  // for (int n=0; n<cshape[0]; n++) {
  //   for (int i=0; i<fmhei; i=i+pshape[0]) {
  //     for (int j=0; j<fmwid; j=j+pshape[1]) {
  //       for (int k=0; k<pshape[0]; k++) {
  //         for (int l=0; l<pshape[1]; l++) {
  //           if (out_eval[n][i+k][j+l]>max[n]) {
  //             max[n] = out_eval[n][i+k][j+l];
  //             index_ref[n][i/pshape[0]][j/pshape[1]][0] = i + k;
  //             index_ref[n][i/pshape[0]][j/pshape[1]][1] = j + l;
  //           }
  //         }
  //       }
  //       max[n]=INT_MIN;
  //     }
  //   }
  // }
  // int gokei=0;
  // //string filename;
  // //FILE *fp;
  // for (int n=0; n<cshape[0]; n++) {
  //   // sprintf(filename, "./eval_lazy/pmap_err/5/data100_%d_%d.dat", n, cshape[0]);
  //   // fp = fopen(filename, "w");
  //   for (int i=0; i<pmhei; i++) {
  //     for (int j=0; j<pmwid; j++) {
  //       if ((index[n][i][j][0] != index_ref[n][i][j][0]) || (index[n][i][j][1] != index_ref[n][i][j][1])) gokei++;
  //         // printf(" %2d ", ((index[n][i][j][0]-i*pshape[0])*pshape[1]+(index[n][i][j][1]-j*pshape[1]))-((index_ref[n][i][j][0]-i*pshape[0])*pshape[1]+(index_ref[n][i][j][1]-j*pshape[1])));
  //       // if ((index[n][i][j][0] != index_ref[n][i][j][0]) || (index[n][i][j][1] != index_ref[n][i][j][1])) fprintf(fp, "1\n");
  //       // else fprintf(fp, "0\n");
  //       // fprintf(fp, "%d\n", output[n][i][j]/*-out_eval[n][index_ref[n][i][j][0]][index_ref[n][i][j][1]]*/);
  //     }
  //     // printf("\n");
  //   }
  //   // fclose(fp);
  //   // printf("\n");
  // }
  // printf("%d: %f\n", cshape[0], (double)gokei/(cshape[0]*pmhei*pmwid));
}

#endif
