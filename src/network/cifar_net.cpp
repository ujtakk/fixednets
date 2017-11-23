#ifdef _CIFAR_HPP_

#include <limits>

template <typename T>
CifarNet<T>::CifarNet()
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
CifarNet<T>::~CifarNet()
{
}

template <typename T>
void CifarNet<T>::Load(std::string path)
{
  conv1.load(path+"/wb_1");
  conv2.load(path+"/wb_2");
  conv3.load(path+"/wb_3");
  conv4.load(path+"/wb_4");
  conv5.load(path+"/wb_5");
  full6.load(path+"/wb_6");
  full7.load(path+"/wb_7");
}

template <typename T>
void CifarNet<T>::Save(std::string path)
{
}

template <typename T>
void CifarNet<T>::Forward(std::string data)
{
}

template <typename T>
void CifarNet<T>::Backward(int label)
{
}

template <typename T>
void CifarNet<T>::Update()
{
}

template <typename T>
int CifarNet<T>::calc(std::string data)
{
  load_txt(input, data);

  conv1.forward(fmap1, input);
  pool1.forward(pmap1, fmap1);
  relu1.forward(amap1, pmap1);
  conv2.forward(fmap2, amap1);
  pool2.forward(pmap2, fmap2);
  relu2.forward(amap2, pmap2);
  conv3.forward(fmap3, amap2);
  pool3.forward(pmap3, fmap3);
  relu3.forward(amap3, pmap3);
  conv4.forward(fmap4, amap3);
  pool4.forward(pmap4, fmap4);
  relu4.forward(amap4, pmap4);
  conv5.forward(fmap5, amap4);
  pool5.forward(pmap5, fmap5);
  relu5.forward(amap5, pmap5);

  flatten(amap5_flat, amap5);

  full6.forward(hunit1, amap5_flat);
  relu6.forward(aunit1, hunit1);

  full7.forward(output, aunit1);

  //output4.forward(output);

  return classify(output);
}

#endif
