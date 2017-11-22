#ifdef _CIFAR_HPP_

#include <limits>

template <typename T>
CIFAR<T>::CIFAR()
#ifdef _LAZY
  : convpool1{N_F1,    3, FHEI, FWID, PHEI, PWID}
  , convpool2{N_F2, N_F1, FHEI, FWID, PHEI, PWID}
  , convpool3{N_F3, N_F2, FHEI, FWID, PHEI, PWID}
  , convpool4{N_F4, N_F3, FHEI, FWID, PHEI, PWID}
  , convpool5{N_F5, N_F4, FHEI, FWID, PHEI, PWID}
#else
  : conv1{N_F1,    3, FHEI, FWID, 1, (FHEI-1)/2}
  , conv2{N_F2, N_F1, FHEI, FWID, 1, (FHEI-1)/2}
  , conv3{N_F3, N_F2, FHEI, FWID, 1, (FHEI-1)/2}
  , conv4{N_F4, N_F3, FHEI, FWID, 1, (FHEI-1)/2}
  , conv5{N_F5, N_F4, FHEI, FWID, 1, (FHEI-1)/2}
  , pool1{PHEI, PWID, 2}
  , pool2{PHEI, PWID, 2}
  , pool3{PHEI, PWID, 2}
  , pool4{PHEI, PWID, 2}
  , pool5{PHEI, PWID, 2}
#endif
  , full6{N_H1, N_F5*pm5hei*pm5wid}
  , full7{LABEL, N_H1}
{
  input = zeros<T>(3, IMHEI, IMWID);

  fmap1 = zeros<T>(N_F1, IMHEI, IMWID);
  pmap1 = zeros<T>(N_F1, pm1hei, pm1wid);
  amap1 = zeros<T>(N_F1, pm1hei, pm1wid);

  fmap2 = zeros<T>(N_F2, pm1hei, pm1wid);
  pmap2 = zeros<T>(N_F2, pm2hei, pm2wid);
  amap2 = zeros<T>(N_F2, pm2hei, pm2wid);

  fmap3 = zeros<T>(N_F3, pm2hei, pm2wid);
  pmap3 = zeros<T>(N_F3, pm3hei, pm3wid);
  amap3 = zeros<T>(N_F3, pm3hei, pm3wid);

  fmap4 = zeros<T>(N_F4, pm3hei, pm3wid);
  pmap4 = zeros<T>(N_F4, pm4hei, pm4wid);
  amap4 = zeros<T>(N_F4, pm4hei, pm4wid);

  fmap5 = zeros<T>(N_F5, pm4hei, pm4wid);
  pmap5 = zeros<T>(N_F5, pm5hei, pm5wid);
  amap5 = zeros<T>(N_F5, pm5hei, pm5wid);

  amap5_flat = zeros<T>(N_F5*pm5hei*pm5wid);

  hunit1 = zeros<T>(N_H1);
  aunit1 = zeros<T>(N_H1);
  output = zeros<T>(LABEL);
}

template <typename T>
CIFAR<T>::~CIFAR()
{
}

template <typename T>
void CIFAR<T>::Load(std::string path)
{
#ifdef _LAZY
  convpool1.load(path+"/wb_1");
  convpool2.load(path+"/wb_2");
  convpool3.load(path+"/wb_3");
  convpool4.load(path+"/wb_4");
  convpool5.load(path+"/wb_5");
#else
  conv1.load(path+"/wb_1");
  conv2.load(path+"/wb_2");
  conv3.load(path+"/wb_3");
  conv4.load(path+"/wb_4");
  conv5.load(path+"/wb_5");
#endif
  full6.load(path+"/wb_6");
  full7.load(path+"/wb_7");
}

template <typename T>
void CIFAR<T>::Save(std::string path)
{
}

template <typename T>
void CIFAR<T>::Forward(std::string data)
{
}

template <typename T>
void CIFAR<T>::Backward(int label)
{
}

template <typename T>
void CIFAR<T>::Update()
{
}

template <typename T>
int CIFAR<T>::calc(std::string data, int which, int amount)
{
  int number = -1;
  int temp = std::numeric_limits<int>::min();

  load_txt(input, data);

#ifdef _LAZY
  convpool1.forward(input, pmap1, which, amount);
  relu1.forward(pmap1, amap1);
  convpool2.forward(amap1, pmap2, which, amount);
  relu2.forward(pmap2, amap2);
  convpool3.forward(amap2, pmap3, which, amount);
  relu3.forward(pmap3, amap3);
  convpool4.forward(amap3, pmap4, which, amount);
  relu4.forward(pmap4, amap4);
  convpool5.forward(amap4, pmap5, which, amount);
  relu5.forward(pmap5, amap5);
#else
  conv1.forward(input, fmap1);
  pool1.forward(fmap1, pmap1);
  relu1.forward(pmap1, amap1);
  conv2.forward(amap1, fmap2);
  pool2.forward(fmap2, pmap2);
  relu2.forward(pmap2, amap2);
  conv3.forward(amap2, fmap3);
  pool3.forward(fmap3, pmap3);
  relu3.forward(pmap3, amap3);
  conv4.forward(amap3, fmap4);
  pool4.forward(fmap4, pmap4);
  relu4.forward(pmap4, amap4);
  conv5.forward(amap4, fmap5);
  pool5.forward(fmap5, pmap5);
  relu5.forward(pmap5, amap5);
#endif

  flatten(amap5, amap5_flat);

  full6.forward(amap5_flat, hunit1);
  relu6.forward(hunit1, aunit1);

  full7.forward(aunit1, output);

  //output4.forward(output);

  for(int i=0;i<LABEL;i++) {
    if(temp < output[i]) {
      temp = output[i];
      number = i;
    }
  }

  return number;
}

#endif
