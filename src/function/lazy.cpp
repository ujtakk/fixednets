#ifdef _LAZY_HPP_

#include <limits>

void lazy(
  Mat3D<int>& input,
  Mat4D<int>& iw,
  Mat1D<int>& ib,
  Mat3D<int>& output,
  const int out_c, const int in_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
)
{
  const int ihei = input[0].size();
  const int iwid = input[0][0].size();
  const int fmhei = ihei-fhei+1;
  const int fmwid = iwid-fwid+1;
  const int pmhei = fmhei / phei;
  const int pmwid = fmwid / pwid;

  Mat1D<int> max(out_c, std::numeric_limits<int>::min());
  Mat1D<int> x(out_c, 0);
  Mat1D<int> y(out_c, 0);
  Mat1D<int> p(out_c, 0);
  Mat1D<int> s(out_c, 0);
  Mat3D<int> sum;
  Mat3D<int> total;

  Mat3D<int> out_trunc;

  Mat4D<int> index;

  sum = zeros<int>(out_c, fmhei, fmwid);
  total = zeros<int>(out_c, pmhei, pmwid);

  out_trunc = zeros<int>(out_c, fmhei, fmwid);

  index = zeros<int>(out_c, pmhei, pmwid, 2);

  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n = 0; n < out_c; n++) {
    for (int m = 0; m < in_c; m++) {
      for (int i = fhei-1; i < ihei; i++) {
        for (int j = fwid-1; j < iwid; j++) {
          for (int k = 0; k < fhei; k++) {
            for (int l = 0; l < fwid; l++) {
              //p[n] = (input[m][i-k][j-l] * iw[n][m][k][l]) >> 8;
              if (iw[n][m][k][l] >= 0) p[n] = input[m][i-k][j-l];
              else if (iw[n][m][k][l] < 0) p[n] = -input[m][i-k][j-l];
              s[n] += p[n];
            }
          }
          sum[n][i-fhei+1][j-fwid+1] += s[n];
          s[n] = 0;
        }
      }
    }

    for (int k = 0; k < fmhei; k++) {
      for (int l = 0; l < fmwid; l++) {
        out_trunc[n][k][l] = sum[n][k][l];
        sum[n][k][l] = 0;
      }
    }
  }

  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n = 0; n < out_c; n++) {
    for (int i = 0; i < pmhei; i++) {
      for (int j = 0; j < pmwid; j++) {
        for (int k = 0; k < phei; k++) {
          for (int l = 0; l < pwid; l++) {
            if (out_trunc[n][i*phei+k][j*pwid+l] > max[n]) {
              max[n] = out_trunc[n][i*phei+k][j*pwid+l];
              index[n][i][j][0] = i*phei + k;
              index[n][i][j][1] = j*pwid + l;
            }
          }
        }
        max[n]=std::numeric_limits<int>::min();
      }
    }
  }

  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n = 0; n < out_c; n++) {
    for (int m = 0; m < in_c; m++) {
      for (int i = 0; i < pmhei; i++) {
        for (int j = 0; j < pmwid; j++) {
          x[n] = index[n][i][j][0] + fhei - 1;
          y[n] = index[n][i][j][1] + fwid - 1;
          for (int k = 0; k < fhei; k++) {
            for (int l = 0; l < fwid; l++) {
              p[n] = (input[m][x[n]-k][y[n]-l] * iw[n][m][k][l]) >> 8;
              s[n] += p[n];
            }
          }
          total[n][i][j] += s[n];
          s[n] = 0;
        }
      }
    }
    for (int i = 0; i < pmhei; i++) {
      for (int j = 0; j < pmwid; j++) {
        output[n][i][j] = total[n][i][j] + ib[n];
        total[n][i][j] = 0;
      }
    }
  }
}

void appconv(
  Mat3D<int>& input,
  Mat4D<int>& iw,
  Mat3D<int>& out_trunc,
  const int ihei, const int iwid,
  const int out_c, const int in_c,
  const int fhei, const int fwid
)
{
  const int fmhei = ihei-fhei+1;
  const int fmwid = iwid-fwid+1;

  Mat1D<int> p(out_c, 0);
  Mat1D<int> s(out_c, 0);
  Mat3D<int> sum;
  sum = zeros<int>(out_c, fmhei, fmwid);

  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n = 0; n < out_c; n++) {
    for (int m = 0; m < in_c; m++) {
      for (int i = fhei-1; i < ihei; i++) {
        for (int j = fwid-1; j < iwid; j++) {
          for (int k = 0; k < fhei; k++) {
            for (int l = 0; l < fwid; l++) {
              //p[n] = (input[m][i-k][j-l] * iw[n][m][k][l]) >> 8;
              if (iw[n][m][k][l] >= 0) p[n] = input[m][i-k][j-l];
              else if (iw[n][m][k][l] < 0) p[n] = -input[m][i-k][j-l];
              s[n] += p[n];
            }
          }
          sum[n][i-fhei+1][j-fwid+1] += s[n];
          s[n] = 0;
        }
      }
    }

    for (int k = 0; k < fmhei; k++)
    {
      for (int l = 0; l < fmwid; l++)
      {
        out_trunc[n][k][l] = sum[n][k][l];
        sum[n][k][l] = 0;
      }
    }
  }
}

void maxindex(
  Mat3D<int>& out_trunc,
  Mat4D<int>& index,
  const int ihei, const int iwid,
  const int out_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
)
{
  const int fmhei = ihei-fhei+1;
  const int fmwid = iwid-fwid+1;
  const int pmhei = fmhei / phei;
  const int pmwid = fmwid / pwid;
  Mat1D<int> max(out_c, std::numeric_limits<int>::min());

  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n = 0; n < out_c; n++) {
    for (int i = 0; i < pmhei; i++) {
      for (int j = 0; j < pmwid; j++) {
        for (int k = 0; k < phei; k++) {
          for (int l = 0; l < pwid; l++) {
            if (out_trunc[n][i*phei+k][j*pwid+l] > max[n]) {
              max[n] = out_trunc[n][i*phei+k][j*pwid+l];
              index[n][i][j][0] = i*phei + k;
              index[n][i][j][1] = j*pwid + l;
            }
          }
        }
        max[n]=std::numeric_limits<int>::min();
      }
    }
  }
}

void preconv(
  Mat4D<int>& index,
  Mat3D<int>& input,
  Mat4D<int>& iw,
  Mat1D<int>& ib,
  Mat3D<int>& output,
  const int ihei, const int iwid,
  const int out_c, const int in_c,
  const int fhei, const int fwid,
  const int phei, const int pwid
)
{
  const int fmhei = ihei-fhei+1;
  const int fmwid = iwid-fwid+1;
  const int pmhei = fmhei / phei;
  const int pmwid = fmwid / pwid;

  Mat1D<int> p(out_c, 0);
  Mat1D<int> s(out_c, 0);
  Mat1D<int> x(out_c, 0);
  Mat1D<int> y(out_c, 0);
  Mat3D<int> total;
  total = zeros<int>(out_c, pmhei, pmwid);

  #ifdef _openmp
  #pragma omp parallel for
  #endif
  for (int n = 0; n < out_c; n++) {
    for (int m = 0; m < in_c; m++) {
      for (int i = 0; i < pmhei; i++) {
        for (int j = 0; j < pmwid; j++) {
          x[n] = index[n][i][j][0] + fhei - 1;
          y[n] = index[n][i][j][1] + fwid - 1;
          for (int k = 0; k < fhei; k++) {
            for (int l = 0; l < fwid; l++) {
              p[n] = (input[m][x[n]-k][y[n]-l] * iw[n][m][k][l]) >> 8;
              s[n] += p[n];
            }
          }
          total[n][i][j] += s[n];
          s[n] = 0;
        }
      }
    }
    for (int i = 0; i < pmhei; i++) {
      for (int j = 0; j < pmwid; j++) {
        output[n][i][j] = total[n][i][j] + ib[n];
        total[n][i][j] = 0;
      }
    }
  }
}

#endif
