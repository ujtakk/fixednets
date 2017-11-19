#ifdef _ERROR_HPP_

void load_error1(Mat2D<int>& etable, int volt)
{
  FILE *fp;
  char filename[256];
  int i = 0;

  sprintf(filename,"/home/work/takau/model/1.%dv/emodel1.dat",volt);

  if ((fp = fopen(filename,"r")) == NULL)
    throw "failed load_error1";

  while (fscanf(fp,"%d %d %d", &etable[i][0], &etable[i][1], &etable[i][2]) == 3) {
    i++;
  }

  fclose(fp);
}

void load_error2(Mat2D<int>& etable, int volt)
{
  FILE *fp;
  char filename[256];
  int i = 0;

  sprintf(filename,"/home/work/takau/model/1.%dv/emodel2.dat",volt);

  if ((fp = fopen(filename,"r")) == NULL)
    throw "failed load_error2";

  while (fscanf(fp,"%d %d %d", &etable[i][0], &etable[i][1], &etable[i][2]) == 3) {
    i++;
  }

  fclose(fp);
}

int rand_error1(int output, const int N_EM1, Mat2D<int>& etable)
{
  int error = 0;
  int min   = 0;
  int max   = 0;

  for (int i=0;i<N_EM1-1;i++) {
    if (output == 0) {
      if ((etable[i+1][0] == 0) && (etable[i][0] < 0)) {
        min = i+1;
      }
      if ((etable[i][0] == 0) && ((etable[i+1][0] > 0) || (etable[i+1][1] == 0))) {
        max = i;
        break;
      }
    }
    else if (output < 0) {
      if ((etable[i+1][0] > output-120) && (etable[i][0] <= output-120)) {
        min = i+1;
      }
      if ((etable[i][0] <= output) && ((etable[i+1][0] > output) || (etable[i+1][1] == 0))) {
        max = i;
        break;
      }
    }
    else if (output > 0) {
      if (etable[i][0] <= 0) {
        if ((etable[i+1][0] > 0) && (etable[i+1][0] > output-120)) {
          min = i+1;
        }
      }
      if (etable[i][0] > 0) {
        if ((etable[i+1][0] > output-120) && (etable[i][0] <= output-120)) {
          min = i+1;
        }
      }
      if ((etable[i][0] <= output) && ((etable[i+1][0] > output) || (etable[i+1][1] == 0))) {
        max = i;
        break;
      }
    }
  }

  if (etable[max][1] == 0)
    return output;

  int r = rand() % etable[max][1];

  for (int i = min; i < max+1; i++) {
    error = etable[i][2];

    if (etable[i][1]>r)
      break;
  }

  output += error;
  return output;
}

//output is scalar
int rand_error2(int output, const int N_EM2, Mat2D<int>& etable, int& flag)
{
  int error = 0;
  int min   = 0;
  int max   = 0;
  //int err_flag = 0;

  for (int i = 0; i < N_EM2-1; i++) {
    if (output == 0) {
      if ((etable[i+1][0] == 0) && (etable[i][0] < 0)) {
        min = i+1;
      }
      if ((etable[i][0] == 0) && ((etable[i+1][0] > 0) || (etable[i+1][1] == 0))) {
        max = i;
        break;
      }
    }
    else if (output < 0) {
      if ((etable[i+1][0] > output-75) && (etable[i][0] <= output-75)) {
        min = i+1;
      }
      if ((etable[i][0] <= output) && ((etable[i+1][0] > output) || (etable[i+1][1] == 0))) {
        max = i;
        break;
      }
    }
    else if (output > 0) {
      if (etable[i][0] <= 0) {
        if ((etable[i+1][0] > 0) && (etable[i+1][0] > output-75)) {
          min = i+1;
        }
      }
      if (etable[i][0] > 0) {
        if ((etable[i+1][0] > output-75) && (etable[i][0] <= output-75)) {
          min = i+1;
        }
      }
      if ((etable[i][0] <= output) && ((etable[i+1][0] > output) || (etable[i+1][1] == 0))) {
        max = i;
        break;
      }
    }
  }

  if (etable[max][1] == 0)
    return output;

  int r = rand() % etable[max][1];

  for (int i = min; i < max+1; i++) {
    error = etable[i][2];

    if (error != 0)
      flag = 1;

    if (etable[i][1] > r)
      break;
  }

  //printf("%d : %d\n", output, error);
  output += error;
  return output;
}

#endif
