#ifdef _CONV_HPP_

#include "function.hpp"

/*convolution image to feature map*/
void conv_error1(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM1,
  Mat2D<int>& etable
)
{
  int pro;
  int sum = 0;

  for (int i = 0; i < ihei-fhei+1; i++) {
    for (int j = 0; j < iwid-fwid+1; j++) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          pro = (input[i+k][j+l] * fweight[k][l]) >> 8;
          sum += pro;
        }
      }
      fmap[i][j] = sum;
      fmap[i][j] = rand_error1(fmap[i][j], N_EM1, etable);
      sum = 0;
    }
  }
}

void conv_error1_bias(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM1,
  int bias, double prob
)
{
  int pro;
  int sum = 0;

  for (int i = 0; i < ihei-fhei+1; i++) {
    for (int j = 0; j < iwid-fwid+1; j++) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          pro = (input[i+k][j+l] * fweight[k][l]) >> 8;
          pro = approx(pro, bias, prob);
          sum += pro;
        }
      }
      fmap[i][j] = sum;
      sum = 0;
    }
  }
}

int conv_error2(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  Mat2D<int>& etable
)
{
  int pro;
  int sum = 0;
  int err_flag = 0;

  for (int i = 0; i < ihei-fhei+1; i++) {
    for (int j = 0; j < iwid-fwid+1; j++) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          pro = (input[i+k][j+l] * fweight[k][l]) >> 8;
          sum += pro;
        }
      }
      fmap[i][j] = sum;
      fmap[i][j] = rand_error2(fmap[i][j],N_EM2,etable,err_flag);
      sum = 0;
    }
  }

  return err_flag;
}

int conv_error2_bias(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  int bias, double prob
)
{
  int pro;
  int sum = 0;
  int err_flag = 0;

  for (int i = 0; i < ihei-fhei+1; i++) {
    for (int j = 0; j < iwid-fwid+1; j++) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          pro = (input[i+k][j+l] * fweight[k][l]) >> 8;
          pro = approx(pro, bias, prob);
          sum += pro;
        }
      }
      fmap[i][j] = sum;
      sum = 0;
    }
  }

  return err_flag;
}

/*convolution feature map to feature map*/
void fm_fm_e(Mat3D<int>& infm, Mat3D<int>& outfm, Mat4D<int>& fweight,
  const int n_in, const int n_out, const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  Mat2D<int>& etable
)
{
  Mat3D<int> sum;

  sum = zeros<int>(n_out,ihei-fhei+1,iwid-fwid+1);

  for (int i = 0; i < n_out; i++) {
    for (int j = 0; j < n_in; j++) {
      conv_error2(infm[j],fweight[j][i],sum[i],ihei,iwid,fhei,fwid,N_EM2,etable);
      for (int k = 0; k < ihei-fhei+1; k++) {
        for (int l = 0; l < iwid-fwid+1; l++) {
          outfm[i][k][l] += sum[i][k][l];
        }
      }
    }
  }

  //ifree_3d(sum,n_out,ihei-fhei+1);
}

void fm_fm_bias(Mat3D<int>& infm, Mat3D<int>& outfm, Mat4D<int>& fweight,
  const int n_in, const int n_out, const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  int bias, double prob
)
{
  Mat3D<int> sum;

  sum = zeros<int>(n_out,ihei-fhei+1,iwid-fwid+1);

  for (int i = 0; i < n_out; i++) {
    for (int j = 0; j < n_in; j++) {
      conv_error2_bias(infm[j],fweight[j][i],sum[i],ihei,iwid,fhei,fwid,N_EM2,bias,prob);
      for (int k = 0; k < ihei-fhei+1; k++) {
        for (int l = 0; l < iwid-fwid+1; l++) {
          outfm[i][k][l] += sum[i][k][l];
        }
      }
    }
  }

  //ifree_3d(sum,n_out,ihei-fhei+1);
}

void conv_approx(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int which, int amount
)
{
  int pro = 0;
  int sum = 0;

  for (int i = 4; i < ihei; i++) {
    for (int j = 4; j < iwid; j++) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          if (input[i-k][j-l] * fweight[k][l] >= 0)
            pro = (input[i-k][j-l] * fweight[k][l]) >> 8;
          else
            pro = ((input[i-k][j-l] * fweight[k][l]) >> 8)-1;
          sum = ADD(16, sum, pro, which, amount);
        }
      }
      fmap[i-fhei+1][j-fwid+1] = sum;
      sum = 0;
    }
  }
}

/*convolution feature map to feature map*/
void fm_fm(Mat3D<int>& infm, Mat3D<int>& outfm, Mat4D<int>& fweight,
  const int n_in, const int n_out, const int ihei, const int iwid, const int fhei, const int fwid
)
{
  Mat4D<int> sum;

  sum = zeros<int>(n_out,n_in,ihei-fhei+1,iwid-fwid+1);

  for (int i = 0; i < n_out; i++) {
    for (int j = 0; j < n_in; j++) {
      conv(infm[j],fweight[j][i],sum[i][j],ihei,iwid,fhei,fwid);
      for (int k = 0; k < ihei-fhei+1; k++) {
        for (int l = 0; l < iwid-fwid+1; l++) {
          outfm[i][k][l] += sum[i][j][k][l];
        }
      }
    }
  }
}

void fm_fm_approx(Mat3D<int>& infm, Mat3D<int>& outfm, Mat4D<int>& fweight,
  const int n_in, const int n_out, const int ihei, const int iwid, const int fhei, const int fwid,
  int which, int amount
)
{
  Mat4D<int> sum;

  sum = zeros<int>(n_out,n_in,ihei-fhei+1,iwid-fwid+1);

  for (int i = 0; i < n_out; i++) {
    for (int j = 0; j < n_in; j++) {
      conv_approx(infm[j],fweight[j][i],sum[i][j],ihei,iwid,fhei,fwid,which,amount);
      for (int k = 0; k < ihei-fhei+1; k++) {
        for (int l = 0; l < iwid-fwid+1; l++) {
          outfm[i][k][l] += sum[i][j][k][l];
        }
      }
    }
  }
}

/*convolution image to feature map*/
template <typename T>
void conv(Mat2D<T>& input, Mat2D<T>& fweight, Mat2D<T>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
)
{
  int pro = 0;
  T sum = 0;

  for (int i = fhei-1; i < ihei; i++) {
    for (int j = fwid-1; j < iwid; j++) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          pro = input[i-k][j-l] * fweight[k][l];
          if (pro >= 0)
            sum += pro / Q_OFFSET<T>;
          else
            sum += pro / Q_OFFSET<T> - 1;
        }
      }
      fmap[i-fhei+1][j-fwid+1] = sum;
      sum = 0;
    }
  }
}

template <typename T>
void conv_plus(Mat2D<T>& input, Mat2D<T>& fweight, Mat2D<T>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
)
{
  int pro = 0;
  T sum = 0;

  for (int i = 0; i < ihei-fhei+1; i++) {
    for (int j = 0; j < iwid-fwid+1; j++) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          pro = input[i+k][j+l] * fweight[k][l];
          if (pro >= 0)
            sum += pro / Q_OFFSET<T>;
          else
            sum += pro / Q_OFFSET<T> - 1;
        }
      }
      fmap[i][j] = sum;
      sum = 0;
    }
  }
}

void conv_plus_bi(Mat2D<int>& input, Mat2D<int>& fweight, Mat2D<int>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid
)
{
  int pro = 0;
  int sum = 0;

  for (int i = 0; i < ihei-fhei+1; i++) {
    for (int j = 0; j < iwid-fwid+1; j++) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          if (fweight[k][l] >= 0)
            pro = input[i+k][j+l];
          else
            pro = -input[i+k][j+l];
          sum += pro;
        }
      }
      fmap[i][j] = sum;
      sum = 0;
    }
  }
}
template <typename T>
void conv_plus_bi(Mat2D<T>& input, Mat2D<T>& fweight, Mat2D<T>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int stride, int pad
)
{
  T pro = 0;
  T sum = 0;

  Mat2D<T> padded;
  padded = zeros<T>(ihei+2*pad,iwid+2*pad);

  for (int i = 0; i < ihei; i++)
    for (int j = 0; j < iwid; j++)
      padded[i+pad][j+pad] = input[i][j];

  for (int i = 0; i < ihei+2*pad-fhei+1; i+=stride) {
    for (int j = 0; j < iwid+2*pad-fwid+1; j+=stride) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          if (fweight[k][l] >= 0)
            pro = padded[i+k][j+l];
          else
            pro = -padded[i+k][j+l];
          sum += pro;
        }
      }
      fmap[i/stride][j/stride] = sum;
      sum = 0;
    }
  }
}


template <typename T>
void conv_plus_pad(Mat2D<T>& input, Mat2D<T>& fweight, Mat2D<T>& fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int stride, int pad
)
{
  int pro = 0;
  T sum = 0;

  Mat2D<T> padded;
  padded = zeros<T>(ihei+2*pad,iwid+2*pad);

  for (int i = 0; i < ihei; i++)
    for (int j = 0; j < iwid; j++)
      padded[i+pad][j+pad] = input[i][j];

  for (int i = 0; i < ihei+2*pad-fhei+1; i+=stride) {
    for (int j = 0; j < iwid+2*pad-fwid+1; j+=stride) {
      for (int k = 0; k < fhei; k++) {
        for (int l = 0; l < fwid; l++) {
          pro = padded[i+k][j+l] * fweight[k][l];
          if (pro >= 0)
            sum += pro / Q_OFFSET<T>;
          else
            sum += pro / Q_OFFSET<T> - 1;
        }
      }
      fmap[i/stride][j/stride] = sum;
      sum = 0;
    }
  }
}

template <typename T>
void conv_plus_pad(Mat3D<T>& input, Mat4D<T>& weight, Mat3D<T>& output,
                   int stride, int pad)
{
  const int n_out = output.size();

  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  const int fil_h = weight[0][0].size();
  const int fil_w = weight[0][0][0].size();

  const int fea_h = in_h - fil_h + stride + 2*pad;
  const int fea_w = in_w - fil_w + stride + 2*pad;

  // TODO: add static_assert

  Mat3D<T> padded = zeros<T>(n_in, in_h+2*pad, in_w+2*pad);

  for (int m = 0; m < n_in; ++m)
    for (int i = 0; i < in_h; ++i)
      for (int j = 0; j < in_w; ++j)
        padded[m][i+pad][j+pad] = input[m][i][j];

  // #ifdef _OPENMP
  // #pragma omp parallel for
  // #endif
  for (int n = 0; n < n_out; ++n) {
    for (int m = 0; m < n_in; ++m) {
      for (int i = 0; i < fea_h; i+=stride) {
        for (int j = 0; j < fea_w; j+=stride) {
          for (int k = 0; k < fil_h; ++k)
            for (int l = 0; l < fil_w; ++l)
              output[n][i/stride][j/stride] +=
                mul(padded[m][i+k][j+l], weight[n][m][k][l]);
        }
      }
    }
  }
}

#include <cstring>
template <typename T>
void conv_gemm(Mat3D<T>& input, Mat4D<T>& weight, Mat3D<T>& output,
               int stride, int pad)
{
  const int n_out = output.size();

  const int n_in = input.size();
  const int in_h = input[0].size();
  const int in_w = input[0][0].size();

  const int fil_h = weight[0][0].size();
  const int fil_w = weight[0][0][0].size();

  const int fea_h = in_h - fil_h + stride + 2*pad;
  const int fea_w = in_w - fil_w + stride + 2*pad;

  // TODO: add static_assert

  Mat3D<T> padded = zeros<T>(n_in, in_h+2*pad, in_w+2*pad);

  for (int m = 0; m < n_in; ++m)
    for (int i = 0; i < in_h; ++i)
      memmove(&padded[m][i+pad][pad], &input[m][i][0], in_w*sizeof(T));
      // for (int j = 0; j < in_w; ++j)
      //   padded[m][i+pad][j+pad] = input[m][i][j];

  for (int n = 0; n < n_out; ++n) {
    for (int m = 0; m < n_in; ++m) {
      for (int i = 0; i < fea_h; i+=stride) {
        for (int j = 0; j < fea_w; j+=stride) {
          for (int k = 0; k < fil_h; ++k)
            for (int l = 0; l < fil_w; ++l)
              output[n][i/stride][j/stride] +=
                mul(padded[m][i+k][j+l], weight[n][m][k][l]);
        }
      }
    }
  }
}

#endif
