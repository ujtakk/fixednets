#ifdef _ALEXNET_HPP_

#include <iostream>

template <typename T>
AlexNet<T>::AlexNet()
#if defined _EAGER
  : conv1{N_F1,    3, FSIZE1, FSIZE1, STRID1,            0}
  , conv2{N_F2, N_F1, FSIZE2, FSIZE2,      1, (FSIZE2-1)/2}
  , conv3{N_F3, N_F2, FSIZE3, FSIZE3,      1, (FSIZE3-1)/2}
  , conv4{N_F4, N_F3, FSIZE3, FSIZE3,      1, (FSIZE3-1)/2}
  , conv5{N_F5, N_F4, FSIZE3, FSIZE3,      1, (FSIZE3-1)/2}
  , pool1{PSIZE, PSIZE, STRID2}
  , pool2{PSIZE, PSIZE, STRID2}
  , pool5{PSIZE, PSIZE, STRID2}
#elif defined _LAZY
  //: convpool1{N_F1,    3, FSIZE1, FSIZE1, PSIZE, PSIZE, STRID1, 0, STRID2}
  //, convpool2{N_F2, N_F1, FSIZE2, FSIZE2, PSIZE, PSIZE, 1, (FSIZE2-1)/2, STRID2}
  : conv1{N_F1,    3, FSIZE1, FSIZE1, STRID1,            0}
  , conv2{N_F2, N_F1, FSIZE2, FSIZE2,      1, (FSIZE2-1)/2}
  , conv3{N_F3, N_F2, FSIZE3, FSIZE3,      1, (FSIZE3-1)/2}
  , conv4{N_F4, N_F3, FSIZE3, FSIZE3,      1, (FSIZE3-1)/2}
  , convpool5{N_F5, N_F4, FSIZE3, FSIZE3, PSIZE, PSIZE, 1, (FSIZE3-1)/2, STRID2}
  , pool1{PSIZE, PSIZE, STRID2}
  , pool2{PSIZE, PSIZE, STRID2}
#endif
  , full6{N_H1, N_F5*pm5hei*pm5wid}
  , full7{N_H2, N_H1}
  , full8{LABEL, N_H2}
  , bn1(N_F1)
  , bn2(N_F2)
{
  input = zeros<T>(3, INSIZE, INSIZE);

  fmap1 = zeros<T>(N_F1, fm1hei, fm1wid);
  bmap1 = zeros<T>(N_F1, fm1hei, fm1wid);
  pmap1 = zeros<T>(N_F1, pm1hei, pm1wid);
  amap1 = zeros<T>(N_F1, pm1hei, pm1wid);

  fmap2 = zeros<T>(N_F2, pm1hei, pm1wid);
  bmap2 = zeros<T>(N_F2, pm1hei, pm1wid);
  pmap2 = zeros<T>(N_F2, pm2hei, pm2wid);
  amap2 = zeros<T>(N_F2, pm2hei, pm2wid);

  fmap3 = zeros<T>(N_F3, pm2hei, pm2wid);
  amap3 = zeros<T>(N_F3, pm2hei, pm2wid);

  fmap4 = zeros<T>(N_F4, pm2hei, pm2wid);
  amap4 = zeros<T>(N_F4, pm2hei, pm2wid);

  fmap5 = zeros<T>(N_F5, pm2hei, pm2wid);
  pmap5 = zeros<T>(N_F5, pm5hei, pm5wid);
  amap5 = zeros<T>(N_F5, pm5hei, pm5wid);

  amap5_flat = zeros<T>(N_F5*pm5hei*pm5wid);

  hunit1 = zeros<T>(N_H1);
  aunit1 = zeros<T>(N_H1);
  hunit2 = zeros<T>(N_H2);
  aunit2 = zeros<T>(N_H2);
  output = zeros<T>(LABEL);
}

template <typename T>
AlexNet<T>::~AlexNet()
{
}

template <typename T>
void AlexNet<T>::Load(string path)
{
#if defined _EAGER
  conv1.load(path+"/wb_1");
#elif defined _LAZY
  conv1.load(path+"/wb_1");
  //convpool1.load(filename);
#endif
  std::cout << "conv1 loaded." << std::endl;
  bn1.load(path+"/bn_1");
  std::cout << "bn1 loaded." << std::endl;
#if defined _EAGER
  conv2.load(path+"/wb_2");
#elif defined _LAZY
  conv2.load(path+"/wb_2");
  //convpool2.load(filename);
#endif
  std::cout << "conv2 loaded." << std::endl;
  bn2.load(path+"/bn_2");
  std::cout << "bn2 loaded." << std::endl;
  conv3.load(path+"/wb_3");
  std::cout << "conv3 loaded." << std::endl;
  conv4.load(path+"/wb_4");
  std::cout << "conv4 loaded." << std::endl;
#if defined _EAGER
  conv5.load(path+"/wb_5");
#elif defined _LAZY
  convpool5.load(path+"/wb_5");
#endif
  std::cout << "conv5 loaded." << std::endl;
  full6.load(path+"/wb_6");
  std::cout << "full6 loaded." << std::endl;
  full7.load(path+"/wb_7");
  std::cout << "full7 loaded." << std::endl;
  full8.load(path+"/wb_8");
  std::cout << "full8 loaded." << std::endl;
}

template <typename T>
std::vector<int> AlexNet<T>::calc(string data)
{
  // Top-5 label
  std::vector<int> number(5, -1);

  load_image(data, input);

#if defined _EAGER
  conv1.forward(input, fmap1);
    bn1.forward(fmap1, bmap1);
  pool1.forward(bmap1, pmap1);
  relu1.forward(pmap1, amap1);
  conv2.forward(amap1, fmap2);
    bn2.forward(fmap2, bmap2);
  pool2.forward(bmap2, pmap2);
  relu2.forward(pmap2, amap2);
  conv3.forward(amap2, fmap3);
  relu3.forward(fmap3, amap3);
  conv4.forward(amap3, fmap4);
  relu4.forward(fmap4, amap4);
  conv5.forward(amap4, fmap5);
  pool5.forward(fmap5, pmap5);
  relu5.forward(pmap5, amap5);
#elif defined _LAZY
  const int which = 0;
  const int amount = 0;
  //convpool1.forward(input, pmap1, which, amount);
  conv1.forward(input, fmap1);
    bn1.forward(fmap1, bmap1);
  pool1.forward(bmap1, pmap1);
  relu1.forward(pmap1, amap1);
  //convpool2.forward(amap1, pmap2, which, amount);
  conv2.forward(amap1, fmap2);
    bn2.forward(fmap2, bmap2);
  pool2.forward(bmap2, pmap2);
  relu2.forward(pmap2, amap2);
  conv3.forward(amap2, fmap3);
  relu3.forward(fmap3, amap3);
  conv4.forward(amap3, fmap4);
  relu4.forward(fmap4, amap4);
  convpool5.forward(amap4, pmap5, which, amount);
  relu5.forward(pmap5, amap5);
#endif

  flatten(amap5, amap5_flat, N_F5, pm5hei, pm5wid);

  full6.forward(amap5_flat, hunit1);
  relu6.forward(hunit1, aunit1);
  full7.forward(aunit1, hunit2);
  relu7.forward(hunit2, aunit2);

  full8.forward(aunit2, output);

  //output4.forward(output);

  std::vector<int> index(LABEL);
  iota(index.begin(), index.end(), 0);
  sort(index.begin(), index.end(), [&](int a, int b){
    return output[a] > output[b];
  });
  for (int i = 0; i < 5; i++)
    number[i] = index[i];

  return number;
}

#endif
