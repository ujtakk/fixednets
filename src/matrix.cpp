#ifdef _MATRIX_HPP_

/* memory allocation for each type and dimension */
template <typename T>
Mat1D<T> zeros(int size1)
{
  Mat1D<T> inst;

  inst.resize(size1, 0.0);

  return inst;
}

template <typename T>
Mat2D<T> zeros(int size1, int size2)
{
  Mat2D<T> inst;

  inst.resize(size1);
  for (int i = 0; i < size1; i++) {
    inst[i].resize(size2, 0.0);
  }

  return inst;
}

template <typename T>
Mat3D<T> zeros(int size1, int size2, int size3)
{
  Mat3D<T> inst;

  inst.resize(size1);
  for (int i = 0; i < size1; i++) {
    inst[i].resize(size2);
    for (int j = 0; j < size2; j++) {
      inst[i][j].resize(size3, 0.0);
    }
  }

  return inst;
}

template <typename T>
Mat4D<T> zeros(int size1, int size2, int size3, int size4)
{
  Mat4D<T> inst;

  inst.resize(size1);
  for (int i = 0; i < size1; i++) {
    inst[i].resize(size2);
    for (int j = 0; j < size2; j++) {
      inst[i][j].resize(size3);
      for (int k = 0; k < size3; k++) {
        inst[i][j][k].resize(size4, 0.0);
      }
    }
  }
  return inst;
}

#endif
