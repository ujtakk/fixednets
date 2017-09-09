#ifdef _POOL_HPP_

#include <limits>

/*max pooling*/
void max_pooling(
  Mat2D<int> &fmap, Mat2D<int> &pmap,
  const int fmhei, const int fmwid, const int phei, const int pwid
)
{
  for (int i = 0; i < fmhei; i = i+phei) {
    for (int j = 0; j < fmwid; j = j+pwid) {
      int max = std::numeric_limits<int>::min();

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
static void swap(Mat1D<int> &input,int i,int j)
{
  int temp;

  temp = input[i];
  input[i] = input[j];
  input[j] = temp;
}

void median_pooling(
  Mat2D<int> &fmap, Mat2D<int> &pmap,
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

#endif
