#ifdef _FUNC_HPP_

#include <climits>
#include <cfloat>
#include <random>

#include "error.hpp"
#include "approx.hpp"
#include "fixed_point.hpp"

int approx(int value, int bias, double prob)
{
  int biased = value;
  int rnd_value;
  std::mt19937 mt(10);

  rnd_value = std::abs((int)mt()) % 10000000;

  if ((rnd_value / 10000000.0) < prob)
    biased = biased + bias;

  return biased;
}

/*convolution image to feature map*/
void conv1(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid, const int N_EM1,
  vector<vector<int>> &etable
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

void conv1_bias(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid, const int N_EM1,
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

int conv2(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid, const int N_EM2,
  vector<vector<int>> &etable
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

int conv2_bias(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid, const int N_EM2,
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
void fm_fm_e(
  vector<vector<vector<int>>> &infm,
  vector<vector<vector<int>>> &outfm,
  vector<vector<vector<vector<int>>>> &fweight,
  const int n_in, const int n_out, const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  vector<vector<int>> &etable
)
{
  vector<vector<vector<int>>> sum;

  sum = zeros<int>(n_out,ihei-fhei+1,iwid-fwid+1);

  for (int i = 0; i < n_out; i++) {
    for (int j = 0; j < n_in; j++) {
      conv2(infm[j],fweight[j][i],sum[i],ihei,iwid,fhei,fwid,N_EM2,etable);
      for (int k = 0; k < ihei-fhei+1; k++) {
        for (int l = 0; l < iwid-fwid+1; l++) {
          outfm[i][k][l] += sum[i][k][l];
        }
      }
    }
  }

  //ifree_3d(sum,n_out,ihei-fhei+1);
}

void fm_fm_bias(
  vector<vector<vector<int>>> &infm,
  vector<vector<vector<int>>> &outfm,
  vector<vector<vector<vector<int>>>> &fweight,
  const int n_in, const int n_out, const int ihei, const int iwid,
  const int fhei, const int fwid, const int N_EM2,
  int bias, double prob
)
{
  vector<vector<vector<int>>> sum;

  sum = zeros<int>(n_out,ihei-fhei+1,iwid-fwid+1);

  for (int i = 0; i < n_out; i++) {
    for (int j = 0; j < n_in; j++) {
      conv2_bias(infm[j],fweight[j][i],sum[i],ihei,iwid,fhei,fwid,N_EM2,bias,prob);
      for (int k = 0; k < ihei-fhei+1; k++) {
        for (int l = 0; l < iwid-fwid+1; l++) {
          outfm[i][k][l] += sum[i][k][l];
        }
      }
    }
  }

  //ifree_3d(sum,n_out,ihei-fhei+1);
}

/*convolution image to feature map*/
template <typename T>
void conv(
  vector<vector<T>> &input,
  vector<vector<T>> &fweight,
  vector<vector<T>> &fmap,
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
void conv_plus(
  vector<vector<T>> &input,
  vector<vector<T>> &fweight,
  vector<vector<T>> &fmap,
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

void conv_plus_bi(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
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
void conv_plus_bi(
  vector<vector<T>> &input,
  vector<vector<T>> &fweight,
  vector<vector<T>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int stride, int pad
)
{
  T pro = 0;
  T sum = 0;

  vector<vector<T>> padded;
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
void conv_plus_pad(
  vector<vector<T>> &input,
  vector<vector<T>> &fweight,
  vector<vector<T>> &fmap,
  const int ihei, const int iwid, const int fhei, const int fwid,
  int stride, int pad
)
{
  int pro = 0;
  T sum = 0;

  vector<vector<T>> padded;
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

void conv_approx(
  vector<vector<int>> &input,
  vector<vector<int>> &fweight,
  vector<vector<int>> &fmap,
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
void fm_fm(
  vector<vector<vector<int>>> &infm,
  vector<vector<vector<int>>> &outfm,
  vector<vector<vector<vector<int>>>> &fweight,
  const int n_in, const int n_out, const int ihei, const int iwid, const int fhei, const int fwid
)
{
  vector<vector<vector<vector<int>>>> sum;

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

void fm_fm_approx(
  vector<vector<vector<int>>> &infm,
  vector<vector<vector<int>>> &outfm,
  vector<vector<vector<vector<int>>>> &fweight,
  const int n_in, const int n_out, const int ihei, const int iwid, const int fhei, const int fwid,
  int which, int amount
)
{
  vector<vector<vector<vector<int>>>> sum;

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

/*max pooling*/
void max_pooling(
  vector<vector<int>> &fmap,
  vector<vector<int>> &pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
)
{
  for (int i = 0; i < fmhei; i = i+phei) {
    for (int j = 0; j < fmwid; j = j+pwid) {
      int max = INT_MIN;

      for (int k = 0; k < phei; k++) {
        for (int l = 0; l < pwid; l++) {
          if (fmap[i+k][j+l] > max)
            max = fmap[i+k][j+l];
        }
      }

      pmap[i/phei][j/pwid] = max;
    }
  }
}

/*median pooling*/
void swap(vector<int> &input,int i,int j) {
  int temp;

  temp = input[i];
  input[i] = input[j];
  input[j] = temp;
}

void median_pooling(
  vector<vector<int>> &fmap,
  vector<vector<int>> &pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
)
{
  vector<int> cluster(phei*pwid);
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

/*add bias*/
void add_bias(vector<vector<int>> &input,int bias,int ihei,int iwid)
{
  for (int i = 0; i < ihei; i++)
    for (int j = 0; j < iwid; j++)
      input[i][j] = input[i][j] + bias;
}

/*activation by hinge function*/
void activate(vector<vector<int>> &input, const int ihei, const int iwid)
{
  for (int i = 0; i < ihei; i++)
    for (int j = 0; j < iwid; j++)
      if (input[i][j]<0)
        input[i][j] = 0;
}

void activate_1d(vector<int> &input, const int ilen)
{
  for (int i = 0; i < ilen; i++)
    if (input[i]<0)
      input[i] = 0;
}

/*flatten 3D matrix*/
template <typename T>
void flatten(
  vector<vector<vector<T>>> &matrix,
  vector<T> &array
)
{
  const int mdep = matrix.size();
  const int mhei = matrix[0].size();
  const int mwid = matrix[0][0].size();

  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        array[i*mhei*mwid+j*mwid+k] = matrix[i][j][k];
}

template <typename T>
void flatten(
  vector<vector<vector<T>>> &matrix,
  vector<T> &array,
  const int mdep, const int mhei, const int mwid
)
{
  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        array[i*mhei*mwid+j*mwid+k] = matrix[i][j][k];
}

/*reshape 3D matrix*/
template <typename T>
void reshape(
  vector<T> &array,
  vector<vector<vector<T>>> &matrix
)
{
  const int mdep = matrix.size();
  const int mhei = matrix[0].size();
  const int mwid = matrix[0][0].size();

  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        matrix[i][j][k] = array[i*mhei*mwid+j*mwid+k];
}

template <typename T>
void reshape(
  vector<T> &array,
  vector<vector<vector<T>>> &matrix,
  const int mdep, const int mhei, const int mwid
)
{
  for (int i = 0; i < mdep; i++)
    for (int j = 0; j < mhei; j++)
      for (int k = 0; k < mwid; k++)
        matrix[i][j][k] = array[i*mhei*mwid+j*mwid+k];
}

/*calculation of full connect layer*/
void full_connect(
  vector<int> &input,
  vector<int> &output,
  vector<vector<int>> &weight,
  vector<int> &bias,
  const int ilen, const int olen
)
{
  int pro;
  int sum = 0;
  for (int i = 0; i < olen; i++) {
    for (int j = 0; j < ilen; j++) {
      pro = input[j] * weight[i][j] >> 8;
      sum += pro;
    }
    output[i] = sum + bias[i];
    sum = 0;
  }
}

int softmax(vector<double> &output,int len) {
  double expsum = 0.0;

  for (int i = 0; i < len; i++)
    expsum += exp(output[i]);

  if (std::abs(expsum-0.0) < DBL_EPSILON)
    throw "softmax calculation failed";

  for (int i = 0; i < len; i++)
    output[i] = exp(output[i])/expsum;

  return 0;
}

double mean_1d(vector<double> vec)
{
  double a=0.0;

  for (int i = 0; i < (int)vec.size(); i++)
    a += vec[i];

  return a / (double)vec.size();
}

void lazy(
  vector<vector<vector<int>>> &input,
  vector<vector<vector<vector<int>>>> &iw,
  vector<int> &ib,
  vector<vector<vector<int>>> &output,
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

  vector<int> max(out_c, INT_MIN);
  vector<int> x(out_c, 0);
  vector<int> y(out_c, 0);
  vector<int> p(out_c, 0);
  vector<int> s(out_c, 0);
  vector<vector<vector<int>>> sum;
  vector<vector<vector<int>>> total;

  vector<vector<vector<int>>> out_trunc;

  vector<vector<vector<vector<int>>>> index;

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
        max[n]=INT_MIN;
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
  vector<vector<vector<int>>> &input,
  vector<vector<vector<vector<int>>>> &iw,
  vector<vector<vector<int>>> &out_trunc,
  const int ihei, const int iwid,
  const int out_c, const int in_c,
  const int fhei, const int fwid
)
{
  const int fmhei = ihei-fhei+1;
  const int fmwid = iwid-fwid+1;

  vector<int> p(out_c, 0);
  vector<int> s(out_c, 0);
  vector<vector<vector<int>>> sum;
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
  vector<vector<vector<int>>> &out_trunc,
  vector<vector<vector<vector<int>>>> &index,
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
  vector<int> max(out_c, INT_MIN);

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
        max[n]=INT_MIN;
      }
    }
  }
}

void preconv(
  vector<vector<vector<vector<int>>>> &index,
  vector<vector<vector<int>>> &input,
  vector<vector<vector<vector<int>>>> &iw,
  vector<int> &ib,
  vector<vector<vector<int>>> &output,
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

  vector<int> p(out_c, 0);
  vector<int> s(out_c, 0);
  vector<int> x(out_c, 0);
  vector<int> y(out_c, 0);
  vector<vector<vector<int>>> total;
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
