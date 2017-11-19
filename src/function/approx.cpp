#ifdef _APPROX_HPP_

int ADD(const int bitwidth, int x, int y, int which, int amount)
{
  bool bx[bitwidth];
  bool by[bitwidth];
  bool bz[bitwidth];

  int z;
  bool Cin=false;
  bool Cout;

  convert(bitwidth, x, bx);
  convert(bitwidth, y, by);

  switch (which) {
    case 0: RCA(bitwidth, bx, by, Cin, bz, Cout); break;
    case 1: AMRCA(bitwidth, bx, by, Cin, bz, Cout, 1, amount); break;
    case 2: AMRCA(bitwidth, bx, by, Cin, bz, Cout, 2, amount); break;
    case 3: AMRCA(bitwidth, bx, by, Cin, bz, Cout, 3, amount); break;
    case 4: AXRCA(bitwidth, bx, by, Cin, bz, Cout, 1, amount); break;
    case 5: AXRCA(bitwidth, bx, by, Cin, bz, Cout, 2, amount); break;
    case 6: AXRCA(bitwidth, bx, by, Cin, bz, Cout, 3, amount); break;
    case 7: LORCA(bitwidth, bx, by, Cin, bz, Cout, amount); break;
    case 8: TruncRCA(bitwidth, bx, by, Cin, bz, Cout, amount); break;
    case 9: ESRCA(bitwidth, bx, by, Cin, bz, Cout, bitwidth-amount); break;
    case 10: ETRCA(bitwidth, bx, by, Cin, bz, Cout, bitwidth-amount); break;
    default:
      throw "this type of adder is not implemented";
      break;
  }

  z = print(bitwidth, bz, bx[bitwidth-1], by[bitwidth-1], Cout);

  return z;
}

long long MULT(const int bitwidth, int x, int y, int radix, bool isbooth, int which, int amount)
{
  bool bx[bitwidth];
  bool by[bitwidth];
  bool bz[2*bitwidth];

  long long z;

  convert(bitwidth, x, bx);
  convert(bitwidth, y, by);

  switch (bitwidth) {
    case 2: R2_Wallace2(bx, by, bz); break;
    case 4: R2_Wallace4(bx, by, bz); break;
    case 8: R2_Wallace8(bx, by, bz); break;
    case 16:
      if (isbooth)
        R4_Booth16(bx, by, bz, which, amount);
      else
        R2_Wallace16(bx, by, bz, which, amount);
      break;
    default:
      throw "multiply for that bitwidth is not implemented";
      break;
  }

  z = print_mult(2*bitwidth, bz, bx[bitwidth-1], by[bitwidth-1]);

  return z;
}

//TODO: Confirm whether or not Cout is needed.
int print(const int bitwidth, bool* A, bool sgnx, bool sgny, bool Cout)
{
  int total = 0;

  for (int i = 0; i < bitwidth-1; i++)
    if (A[i]) total += 1 << i;

  // both 0
  if (!sgnx && !sgny) {
    if (A[bitwidth-1]) total += 1 << (bitwidth-1);
  }
  else if (sgnx && sgny) {
    // both 1
    if (A[bitwidth-1]) total -= 1 << (bitwidth-1);
    else if (Cout) total -= 1 << bitwidth;
  }
  else {
    if (A[bitwidth-1]) total -= 1 << (bitwidth-1);
  }

  return total;
}

long long print_mult(const int bitwidth, bool* A, bool sgnx, bool sgny)
{
  long long total = 0;

  for (int i = 0; i < bitwidth-1; i++)
    if (A[i]) total += 1 << i;

  if (sgnx ^ sgny)
    if (A[bitwidth-1]) total -= (long long)1 << (bitwidth-1);
  // else
  //   if (A[bitwidth-1]) total += 1 << (bitwidth-1);

  return total;
}

void convert(const int bitwidth, int number, bool* A)
{
  int x = number;
  bool S, C;

  if (x < 0) {
    x = -x;
    C = true;

    for (int i = 0; i < bitwidth; i++) {
      if (x % 2)  A[i] = false;
      else        A[i] = true;

      S = A[i] ^ C;
      C = A[i] & C;
      A[i] = S;
      x = x/2;
    }
  }
  else {
    for (int i = 0; i < bitwidth; i++) {
      if (x%2) A[i] = true;
      else A[i] = false;

      x = x/2;
    }
  }
}

void HA(bool A, bool B, bool& S, bool& C)
{
  S = A ^ B;
  C = A & B;
}

void FA(bool A, bool B, bool Cin, bool& S, bool& Cout)
{
  bool s1, c1, c2;

  HA(A, B, s1, c1);
  HA(s1, Cin, S, c2);

  Cout = c1 | c2;
}

void AMA1(bool A, bool B, bool Cin, bool& S, bool& Cout)
{
  if (!A) {
    if (!B) {
      if (!Cin) { S = true; Cout = false; return; }
      else      { S = true; Cout = false; return; }
    }
    else {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = false; Cout = true; return; }
    }
  }
  else {
    if (!B) {
      if (!Cin) { S = true; Cout = false; return; }
      else      { S = false; Cout = true; return; }
    }
    else {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = false; Cout = true; return; }
    }
  }
}

void AMA2(bool A, bool B, bool Cin, bool& S, bool& Cout)
{
  if (!A) {
    if (!B) {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = true; Cout = false; return; }
    }
    else {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = true; Cout = false; return; }
    }
  }
  else {
    if (!B) {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = false; Cout = true; return; }
    }
    else {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = true; Cout = true; return; }
    }
  }
}

void AMA3(bool A, bool B, bool Cin, bool& S, bool& Cout)
{
  if (!A) {
    if (!B) {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = false; return; }
    }
    else {
      if (!Cin) { S = true; Cout = false; return; }
      else      { S = true; Cout = false; return; }
    }
  }
  else {
    if (!B) {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = false; Cout = true; return; }
    }
    else {
      if (!Cin) { S = true; Cout = true; return; }
      else      { S = true; Cout = true; return; }
    }
  }
}

void AXA1(bool A, bool B, bool Cin, bool& S, bool& Cout)
{
  if (!A) {
    if (!B) {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = true; Cout = false; return; }
    }
    else {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = true; Cout = false; return; }
    }
  }
  else {
    if (!B) {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = true; Cout = false; return; }
    }
    else {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = true; Cout = true; return; }
    }
  }
}

void AXA2(bool A, bool B, bool Cin, bool& S, bool& Cout)
{
  if (!A) {
    if (!B) {
      if (!Cin) { S = true; Cout = false; return; }
      else      { S = true; Cout = false; return; }
    }
    else {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = true; return; }
    }
  }
  else {
    if (!B) {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = true; return; }
    }
    else {
      if (!Cin) { S = true; Cout = true; return; }
      else      { S = true; Cout = true; return; }
    }
  }
}

void AXA3(bool A, bool B, bool Cin, bool& S, bool& Cout)
{
  if (!A) {
    if (!B) {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = true; Cout = false; return; }
    }
    else {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = true; return; }
    }
  }
  else {
    if (!B) {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = true; return; }
    }
    else {
      if (!Cin) { S = false; Cout = true; return; }
      else      { S = true; Cout = true; return; }
    }
  }
}

void hinagata(bool A, bool B, bool Cin, bool& S, bool& Cout)
{
  if (!A) {
    if (!B) {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = false; return; }
    }
    else {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = false; return; }
    }
  }
  else {
    if (!B) {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = false; return; }
    }
    else {
      if (!Cin) { S = false; Cout = false; return; }
      else      { S = false; Cout = false; return; }
    }
  }
}

void RCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout)
{
  bool c[bitwidth];

  for (int i = 0; i < bitwidth; i++) {
    if (i == 0) FA(A[i], B[i], Cin, S[i], c[i]);
    else        FA(A[i], B[i], c[i-1], S[i], c[i]);
  }

  Cout = c[bitwidth-1];
}

void Cut2RCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout)
{
  bool c[bitwidth];

  for (int i = 0; i < bitwidth; i++) {
    if (i == 0)
      FA(A[i], B[i], Cin, S[i], c[i]);
    else if (i == bitwidth/2)
      FA(A[i], B[i], false, S[i], c[i]);
    else
      FA(A[i], B[i], c[i-1], S[i], c[i]);
  }

  Cout = c[bitwidth-1];
}

void Cut4RCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout)
{
  bool c[bitwidth];

  for (int i = 0; i < bitwidth; i++) {
    if (i == 0)
      FA(A[i], B[i], Cin, S[i], c[i]);
    else if (i == bitwidth/4)
      FA(A[i], B[i], false, S[i], c[i]);
    else if (i == bitwidth/2)
      FA(A[i], B[i], false, S[i], c[i]);
    else if (i == 3*(bitwidth/4))
      FA(A[i], B[i], false, S[i], c[i]);
    else
      FA(A[i], B[i], c[i-1], S[i], c[i]);
  }

  Cout = c[bitwidth-1];
}

void AMRCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout, int which, int amount)
{
  bool c[bitwidth];

  for (int i = 0; i < bitwidth; i++) {
    if (i < amount) {
      if (i == 0) {
        switch (which) {
          case 1: AMA1(A[i], B[i], Cin, S[i], c[i]); break;
          case 2: AMA2(A[i], B[i], Cin, S[i], c[i]); break;
          case 3: AMA3(A[i], B[i], Cin, S[i], c[i]); break;
          default: FA(A[i], B[i], Cin, S[i], c[i]); break;
        }
      }
      else {
        switch (which) {
          case 1: AMA1(A[i], B[i], c[i-1], S[i], c[i]); break;
          case 2: AMA2(A[i], B[i], c[i-1], S[i], c[i]); break;
          case 3: AMA3(A[i], B[i], c[i-1], S[i], c[i]); break;
          default: FA(A[i], B[i], c[i-1], S[i], c[i]); break;
        }
      }
    }
    else {
      if (i == 0) FA(A[i], B[i], Cin, S[i], c[i]);
      else        FA(A[i], B[i], c[i-1], S[i], c[i]);
    }
  }

  Cout = c[bitwidth-1];
}

void AXRCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout, int which, int amount)
{
  bool c[bitwidth];

  for (int i = 0; i < bitwidth; i++) {
    if (i < amount) {
      if (i == 0) {
        switch (which) {
          case 1: AXA1(A[i], B[i], Cin, S[i], c[i]); break;
          case 2: AXA2(A[i], B[i], Cin, S[i], c[i]); break;
          case 3: AXA3(A[i], B[i], Cin, S[i], c[i]); break;
          default: FA(A[i], B[i], Cin, S[i], c[i]); break;
        }
      }
      else {
        switch (which) {
          case 1: AXA1(A[i], B[i], c[i-1], S[i], c[i]); break;
          case 2: AXA2(A[i], B[i], c[i-1], S[i], c[i]); break;
          case 3: AXA3(A[i], B[i], c[i-1], S[i], c[i]); break;
          default: FA(A[i], B[i], c[i-1], S[i], c[i]); break;
        }
      }
    }
    else {
      if (i == 0) FA(A[i], B[i], Cin, S[i], c[i]);
      else        FA(A[i], B[i], c[i-1], S[i], c[i]);
    }
  }

  Cout = c[bitwidth-1];
}

void LORCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout, int amount)
{
  bool c[bitwidth];
  for (int i = 0; i < bitwidth; i++) {
    if (i < amount) {
      if (i == amount-1) {
        S[i] = A[i] | B[i];
        c[i] = A[i] & B[i];
      }
      else {
        S[i] = A[i] | B[i];
        c[i] = false;
      }
    }
    else {
      if (i == 0) FA(A[i], B[i], Cin, S[i], c[i]);
      else FA(A[i], B[i], c[i-1], S[i], c[i]);
    }
  }

  Cout = c[bitwidth-1];
}

void ESRCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout, int length)
{
  int num = bitwidth / length;
  int bottom = bitwidth % length;

  bool a[length];
  bool b[length];
  bool s[length];

  bool c;

  for (int j = 0; j < bottom; j++) {
    a[j] = A[j];
    b[j] = B[j];
  }

  RCA(bottom, a, b, false, s, c);

  for (int j = 0; j < bottom; j++)
    S[j] = s[j];

  for (int i = 0; i < num; i++) {
    for (int j = 0; j < length; j++) {
      a[j] = A[j+i*length+bottom];
      b[j] = B[j+i*length+bottom];
    }

    RCA(length, a, b, false, s, c);

    for (int j = 0; j < length; j++)
      S[j+i*length+bottom] = s[j];

    if (i == 0) Cout = c;
  }
}

void TruncRCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout, int amount)
{
  bool a[bitwidth-amount];
  bool b[bitwidth-amount];
  bool s[bitwidth-amount];

  for (int i = 0; i < bitwidth-amount; i++) {
    a[bitwidth-amount-1-i] = A[bitwidth-1-i];
    b[bitwidth-amount-1-i] = B[bitwidth-1-i];
  }

  RCA(bitwidth-amount, a, b, Cin, s, Cout);

  for (int i = 0; i < bitwidth; i++) {
    if (i < bitwidth-amount)
      S[bitwidth-1-i] = s[bitwidth-amount-1-i];
    else
      S[bitwidth-1-i] = false;
  }
}

void GPRCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout, int length)
{
  bool G[bitwidth];
  bool P[bitwidth];
  bool c[bitwidth+1];

  c[0] = Cin;

  for (int i = 0; i < length; i++) {
    HA(A[i], B[i], P[i], G[i]);

    c[i+1] = G[i] | (P[i] & c[i]);

    S[i] = P[i] ^ c[i];
  }

  Cout = c[bitwidth];
}

void ETRCA(const int bitwidth, bool* A, bool* B, bool Cin, bool* S, bool& Cout, int length)
{
  bool G[bitwidth];
  bool P[bitwidth];
  bool c[bitwidth+1];

  c[0] = Cin;

  for (int i = 0; i < bitwidth; i++) {
    HA(A[i], B[i], P[i], G[i]);

    if (i % length == 0)
      c[i+1] = G[i];
    else
      c[i+1] = G[i] | (P[i] & c[i]);

    S[i] = P[i] ^ c[i];
  }

  Cout = c[bitwidth];
}

void CSA(const int bitwidth, bool* X, bool* Y, bool* Z, bool* S, bool* C, int start, int which, int amount)
{
  C[0] = false;

  for (int i = 0; i < start; i++) {
    S[i] = X[i] | Y[i] | Z[i];
    C[i+1] = false;
  }

  switch (which) {
    case 0:
      for (int i = start; i < start+amount; i++)
        FA(X[i], Y[i], Z[i], S[i], C[i+1]);
      break;

    case 1:
      for (int i = start; i < start+amount; i++)
        AMA1(X[i], Y[i], Z[i], S[i], C[i+1]);
      break;

    case 2:
      for (int i = start; i < start+amount; i++)
        AMA2(X[i], Y[i], Z[i], S[i], C[i+1]);
      break;

    case 3:
      for (int i = start; i < start+amount; i++)
        AMA3(X[i], Y[i], Z[i], S[i], C[i+1]);
      break;

    case 4:
      for (int i = start; i < start+amount; i++)
        AXA1(X[i], Y[i], Z[i], S[i], C[i+1]);
      break;

    case 5:
      for (int i = start; i < start+amount; i++)
        AXA2(X[i], Y[i], Z[i], S[i], C[i+1]);
      break;

    case 6:
      for (int i = start; i < start+amount; i++)
        AXA3(X[i], Y[i], Z[i], S[i], C[i+1]);
      break;

    //3-Input OR Gate
    case 7:
      for (int i = start; i < start+amount; i++) {
        S[i] = X[i] | Y[i] | Z[i];
        C[i+1] = false;
      }
      break;
  }

  for (int i = start+amount; i < bitwidth; i++)
    FA(X[i], Y[i], Z[i], S[i], C[i+1]);
}

void RCA_test(const int bitwidth, bool* X, bool* Y, bool Cin, bool* S, bool& Cout, int start, int which, int amount)
{
  bool C[bitwidth+1];

  C[0] = Cin;

  for (int i = 0; i < start; i++) {
    S[i] = X[i] | Y[i];
    C[i+1] = false;
  }

  switch (which) {
    case 0:
      for (int i = start; i < start+amount; i++)
        FA(X[i], Y[i], C[i], S[i], C[i+1]);
      break;

    case 1:
      for (int i = start; i < start+amount; i++)
        AMA1(X[i], Y[i], C[i], S[i], C[i+1]);
      break;

    case 2:
      for (int i = start; i < start+amount; i++)
        AMA2(X[i], Y[i], C[i], S[i], C[i+1]);
      break;

    case 3:
      for (int i = start; i < start+amount; i++)
        AMA3(X[i], Y[i], C[i], S[i], C[i+1]);
      break;

    case 4:
      for (int i = start; i < start+amount; i++)
        AXA1(X[i], Y[i], C[i], S[i], C[i+1]);
      break;

    case 5:
      for (int i = start; i < start+amount; i++)
        AXA2(X[i], Y[i], C[i], S[i], C[i+1]);
      break;

    case 6:
      for (int i = start; i < start+amount; i++)
        AXA3(X[i], Y[i], C[i], S[i], C[i+1]);
      break;

    // 3-Input OR Gate
    //case 7:
    //  for (int i = start; i < start+amount; i++)
    //  {
    //    S[i] = 
    //  }
    //  break;
  }

  for (int i = start+amount; i < bitwidth; i++)
    FA(X[i], Y[i], C[i], S[i], C[i+1]);

  Cout = C[bitwidth];
}

void R2_Wallace2(bool* A, bool* B, bool* P)
{
  bool PP[4][4] = {{false}};
  bool PS[6][4] = {{false}};
  bool PC[6][5] = {{false}};

  bool Cout;

  bool sgna = A[1];
  bool sgnb = B[1];

  for (int i = 0; i < 2; i++) {
    if (B[i] == true) {
      for (int j = 0; j < 2; j++)
        PP[i][i+j] = A[j];

      for (int j = 2; i+j < 4; j++)
        PP[i][i+j] = sgna;
    }
  }

  if (sgnb == true) {
    for (int i = 2; i < 4; i++)
      for (int j = 0; i+j < 4; j++)
        PP[i][i+j] = A[j];
  }

  CSA(4, PP[0], PP[1], PP[2], PS[0], PC[0], 0, 0, 0);

  CSA(4, PS[0], PC[0], PP[3], PS[1], PC[1], 0, 0, 0);

  RCA(4, PS[1], PC[1], false, P, Cout);
}

void R2_Wallace4(bool* A, bool* B, bool* P)
{
  bool PP[8][8] = {{false}};
  bool PS[6][8] = {{false}};
  bool PC[6][9] = {{false}};

  bool Cout;

  bool sgna = A[3];
  bool sgnb = B[3];

  for (int i = 0; i < 4; i++) {
    if (B[i] == true) {
      for (int j = 0; j < 4; j++)
        PP[i][i+j] = A[j];

      for (int j = 4; i+j < 8; j++)
        PP[i][i+j] = sgna;
    }
  }

  if (sgnb == true) {
    for (int i = 4; i < 8; i++)
      for (int j = 0; i+j < 8; j++)
        PP[i][i+j] = A[j];
  }

  CSA(8, PP[0], PP[1], PP[2], PS[0], PC[0], 0, 0, 0);
  CSA(8, PP[3], PP[4], PP[5], PS[1], PC[1], 0, 0, 0);

  CSA(8, PS[0], PC[0], PS[1], PS[2], PC[2], 0, 0, 0);
  CSA(8, PC[1], PP[6], PP[7], PS[3], PC[3], 0, 0, 0);

  CSA(8, PS[2], PC[2], PS[3], PS[4], PC[4], 0, 0, 0);

  CSA(8, PC[3], PS[4], PC[4], PS[5], PC[5], 0, 0, 0);

  RCA(8, PS[5], PC[5], false, P, Cout);
}

void R2_Wallace8(bool* A, bool* B, bool* P)
{
  bool PP[16][16] = {{false}};
  bool PS[14][16] = {{false}};
  bool PC[14][17] = {{false}};

  bool Cout;

  bool sgna = A[7];
  bool sgnb = B[7];

  for (int i = 0; i < 8; i++) {
    if (B[i] == true) {
      for (int j = 0; j < 8; j++)
        PP[i][i+j] = A[j];

      for (int j = 8; i+j < 16; j++)
        PP[i][i+j] = sgna;
    }
  }

  if (sgnb == true) {
    for (int i = 8; i < 16; i++)
      for (int j = 0; i+j < 16; j++)
        PP[i][i+j] = A[j];
  }

  CSA(16, PP[0], PP[1], PP[2], PS[0], PC[0], 0, 0, 0);
  CSA(16, PP[3], PP[4], PP[5], PS[1], PC[1], 0, 0, 0);
  CSA(16, PP[6], PP[7], PP[8], PS[2], PC[2], 0, 0, 0);
  CSA(16, PP[9], PP[10], PP[11], PS[3], PC[3], 0, 0, 0);
  CSA(16, PP[12], PP[13], PP[14], PS[4], PC[4], 0, 0, 0);

  CSA(16, PS[0], PC[0], PS[1], PS[5], PC[5], 0, 0, 0);
  CSA(16, PC[1], PS[2], PC[2], PS[6], PC[6], 0, 0, 0);
  CSA(16, PS[3], PC[3], PS[4], PS[7], PC[7], 0, 0, 0);

  CSA(16, PS[5], PC[5], PS[6], PS[8], PC[8], 0, 0, 0);
  CSA(16, PC[6], PS[7], PC[7], PS[9], PC[9], 0, 0, 0);

  CSA(16, PS[8], PC[8], PS[9], PS[10], PC[10], 0, 0, 0);
  CSA(16, PC[9], PC[4], PP[15], PS[11], PC[11], 0, 0, 0);

  CSA(16, PS[10], PC[10], PS[11], PS[12], PC[12], 0, 0, 0);

  CSA(16, PS[12], PC[12], PC[11], PS[13], PC[13], 0, 0, 0);

  RCA(16, PS[13], PC[13], false, P, Cout);
}

void R2_Wallace16(bool* A, bool* B, bool* P, int which, int amount)
{
  bool PP[32][32] = {{false}};
  bool PS[30][32] = {{false}};
  bool PC[30][33] = {{false}};

  bool sgna = A[15];
  bool sgnb = B[15];

  bool Cout;

  for (int i = 0; i < 16; i++) {
    if (B[i] == true) {
      for (int j = 0; j < 16; j++)
        PP[i][i+j] = A[j];

      for (int j = 16; i+j < 32; j++)
        PP[i][i+j] = sgna;
    }
  }

  if (sgnb == true) {
    for (int i = 16; i < 32; i++)
      for (int j = 0; i+j < 32; j++)
        PP[i][i+j] = A[j];
  }

  CSA(32, PP[0], PP[1], PP[2], PS[0], PC[0], 1, which, 10);
  CSA(32, PP[3], PP[4], PP[5], PS[1], PC[1], 4, which, 7);
  CSA(32, PP[6], PP[7], PP[8], PS[2], PC[2], 7, which, 4);
  CSA(32, PP[9], PP[10], PP[11], PS[3], PC[3], 10, which, 3);
  CSA(32, PP[12], PP[13], PP[14], PS[4], PC[4], 13, which, 0);
  CSA(32, PP[15], PP[16], PP[17], PS[5], PC[5], 16, which, 0);
  CSA(32, PP[18], PP[19], PP[20], PS[6], PC[6], 19, which, 0);
  CSA(32, PP[21], PP[22], PP[23], PS[7], PC[7], 22, which, 0);
  CSA(32, PP[24], PP[25], PP[26], PS[8], PC[8], 25, which, 0);
  CSA(32, PP[27], PP[28], PP[29], PS[9], PC[9], 28, which, 0);

  CSA(32, PS[0], PC[0], PS[1], PS[10], PC[10], 2, which, 9);
  CSA(32, PC[1], PS[2], PC[2], PS[11], PC[11], 6, which, 5);
  CSA(32, PS[3], PC[3], PS[4], PS[12], PC[12], 11, which, 0);
  CSA(32, PC[4], PS[5], PC[5], PS[13], PC[13], 15, which, 0);
  CSA(32, PS[6], PC[6], PS[7], PS[14], PC[14], 20, which, 0);
  CSA(32, PC[7], PS[8], PC[8], PS[15], PC[15], 24, which, 0);
  CSA(32, PS[9], PC[9], PP[30], PS[16], PC[16], 29, which, 0);

  CSA(32, PS[10], PC[10], PS[11], PS[17], PC[17], 3, which, 4);
  CSA(32, PC[11], PS[12], PC[12], PS[18], PC[18], 9, which, 2);
  CSA(32, PS[13], PC[13], PS[14], PS[19], PC[19], 16, which, 0);
  CSA(32, PC[14], PS[15], PC[15], PS[20], PC[20], 23, which, 0);
  CSA(32, PS[16], PC[16], PP[31], PS[21], PC[21], 30, which, 0);

  CSA(32, PS[17], PC[17], PS[18], PS[22], PC[22], 4, which, 0);
  CSA(32, PC[18], PS[19], PC[19], PS[23], PC[23], 14, which, 0);
  CSA(32, PS[20], PC[20], PS[21], PS[24], PC[24], 24, which, 0);

  CSA(32, PS[22], PC[22], PS[23], PS[25], PC[25], 5, which, 0);
  CSA(32, PC[23], PS[24], PC[24], PS[26], PC[26], 21, which, 0);

  CSA(32, PS[25], PC[25], PS[26], PS[27], PC[27], 6, which, 0);

  CSA(32, PS[27], PC[27], PC[26], PS[28], PC[28], 7, which, 0);

  CSA(32, PS[28], PC[28], PC[21], PS[29], PC[29], 8, which, 0);

  RCA_test(32, PS[29], PC[29], false, P, Cout, 9, which, 0);
}

void R4_Booth16(bool* A, bool* B, bool* P, int which, int amount)
{
  bool PP[8][32] = {{false}};
  bool PS[15][32] = {{false}};
  bool PC[15][33] = {{false}};

  bool As[5][32]; // [0]: O, [1]: A, [2]: -A, [3]: 2A, [4]: -2A
  bool fliped_A[32];
  bool O[32] = {false};
  bool Bs[3] = {false, B[0], B[1]};
  bool test;
  bool Cout;

  int p = 0;

  for (int i = 0; i < 16; i++) {
    As[0][i] = O[i];
    As[1][i] = A[i];
    fliped_A[i] = !A[i];
  }
  for (int i = 16; i < 32; i++) {
    As[0][i] = O[i];
    As[1][i] = A[15];
    fliped_A[i] = !A[15];
  }
  RCA(32, fliped_A, O, true, As[2], test);

  As[3][0] = false;
  As[4][0] = false;
  for (int i = 0; i < 30; i++) {
    As[3][i+1] = As[1][i];
    As[4][i+1] = As[2][i];
  }
  As[3][31] = As[1][31];
  As[4][31] = As[2][31];

  for (int i = 0; i < 8; i++) {
    if ((!Bs[2] && !Bs[1] && !Bs[0]) || (Bs[2] && Bs[1] && Bs[0]))
      p = 0;
    else if ((!Bs[2] && !Bs[1] && Bs[0]) || (!Bs[2] && Bs[1] && !Bs[0]))
      p = 1;
    else if ((Bs[2] && !Bs[1] && Bs[0]) || (Bs[2] && Bs[1] && !Bs[0]))
      p = 2;
    else if (!Bs[2] && Bs[1] && Bs[0])
      p = 3;
    else if (Bs[2] && !Bs[1] && !Bs[0])
      p = 4;

    for (int j = 0; 2*i+j < 32; j++)
      PP[i][2*i+j] = As[p][j];

    if (i < 7) {
      Bs[0] = B[2*i+1];
      Bs[1] = B[2*i+2];
      Bs[2] = B[2*i+3];
    }
  }

  CSA(32, PP[0], PP[1], PP[2], PS[0], PC[0], 2, which, 2);
  CSA(32, PP[3], PP[4], PP[5], PS[1], PC[1], 8, which, 3);

  CSA(32, PS[0], PC[0], PS[1], PS[2], PC[2], 3, which, 3);
  CSA(32, PC[1], PP[6], PP[7], PS[3], PC[3], 12, which, 20);

  CSA(32, PS[2], PC[2], PS[3], PS[4], PC[4], 4, which, 3);

  CSA(32, PS[4], PC[4], PC[3], PS[5], PC[5], 5, which, 0);

  RCA_test(32, PS[5], PC[5], false, P, Cout, 6, which, 0);
}

#endif
