#ifndef _APPROX_HPP_
#define _APPROX_HPP_

#define BITWID 16

int ADD(const int bitwidth, int x, int y, int which, int amount);
long long MULT(const int bitwidth, int x, int y, int radix, bool isbooth, int which, int amount);

int print(const int bitwidth, bool *A, bool sgnx, bool sgny, bool Cout);
long long print_mult(const int bitwidth, bool *A, bool sgnx, bool sgny);
void convert(const int bitwidth, int number, bool *A);

void HA(bool A, bool B, bool &S, bool &C);
void FA(bool A, bool B, bool Cin, bool &S, bool &Cout);
void AMA1(bool A, bool B, bool Cin, bool &S, bool &Cout);
void AMA2(bool A, bool B, bool Cin, bool &S, bool &Cout);
void AXA3(bool A, bool B, bool Cin, bool &S, bool &Cout);
void hinagata(bool A, bool B, bool Cin, bool &S, bool &Cout);

void RCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout);
void Cut2RCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout);
void Cut4RCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout);
void AMRCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout, int which, int amount);
void AXRCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout, int which, int amount);
void LORCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout, int amount);
void ESRCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout, int length);
void TruncRCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout, int amount);
void GPRCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout, int length);
void ETRCA(const int bitwidth, bool *A, bool *B, bool Cin, bool *S, bool &Cout, int length);

void CSA(const int bitwidth, bool *X, bool *Y, bool *Z, bool *S, bool *C, int start, int which, int amount);
void RCA_test(const int bitwidth, bool *X, bool *Y, bool Cin, bool *S, bool &Cout, int start, int which, int amount);
void R2_Wallace2(bool *A, bool *B, bool *P);
void R2_Wallace4(bool *A, bool *B, bool *P);
void R2_Wallace8(bool *A, bool *B, bool *P);
void R2_Wallace16(bool *A, bool *B, bool *P, int which, int amount);
void R4_Booth16(bool *A, bool *B, bool *P, int which, int amount);

#include "approx.cpp"
#endif
