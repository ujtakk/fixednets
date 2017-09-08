#ifndef _ERROR_HPP_
#define _ERROR_HPP_

void load_error1(vector<vector<int>> &etable,int volt);

void load_error2(vector<vector<int>> &etable,int volt);

int rand_error1(int output,const int N_EM1,vector<vector<int>> &etable);

int rand_error2(int output,const int N_EM2,vector<vector<int>> &etable, int &flag);

#include "error.cpp"
#endif
