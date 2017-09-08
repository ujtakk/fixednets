#ifdef _VGG_HPP_

/*
 * TODO: Implement _LAZY Version
 */

template <typename T>
VGGNet<T>::VGGNet()
#if defined _EAGER
  : conv1_1{N_F1,    3, FSIZE, FSIZE, FSTRID, FPAD}
  , conv1_2{N_F1, N_F1, FSIZE, FSIZE, FSTRID, FPAD}
  , pool1{PSIZE, PSIZE, PSTRID}
  , conv2_1{N_F2, N_F1, FSIZE, FSIZE, FSTRID, FPAD}
  , conv2_2{N_F2, N_F2, FSIZE, FSIZE, FSTRID, FPAD}
  , pool2{PSIZE, PSIZE, PSTRID}
  , conv3_1{N_F3, N_F2, FSIZE, FSIZE, FSTRID, FPAD}
  , conv3_2{N_F3, N_F3, FSIZE, FSIZE, FSTRID, FPAD}
  , conv3_3{N_F3, N_F3, FSIZE, FSIZE, FSTRID, FPAD}
  , pool3{PSIZE, PSIZE, PSTRID}
  , conv4_1{N_F4, N_F3, FSIZE, FSIZE, FSTRID, FPAD}
  , conv4_2{N_F4, N_F4, FSIZE, FSIZE, FSTRID, FPAD}
  , conv4_3{N_F4, N_F4, FSIZE, FSIZE, FSTRID, FPAD}
  , pool4{PSIZE, PSIZE, PSTRID}
  , conv5_1{N_F5, N_F4, FSIZE, FSIZE, FSTRID, FPAD}
  , conv5_2{N_F5, N_F5, FSIZE, FSIZE, FSTRID, FPAD}
  , conv5_3{N_F5, N_F5, FSIZE, FSIZE, FSTRID, FPAD}
  , pool5{PSIZE, PSIZE, PSTRID}
#elif defined _LAZY
  : conv1_1{N_F1,    3, FSIZE, FSIZE, FSTRID, FPAD}
  , conv1_2{N_F1, N_F1, FSIZE, FSIZE, FSTRID, FPAD}
  , conv2_1{N_F2, N_F1, FSIZE, FSIZE, FSTRID, FPAD}
  , conv2_2{N_F2, N_F2, FSIZE, FSIZE, FSTRID, FPAD}
  , conv3_1{N_F3, N_F2, FSIZE, FSIZE, FSTRID, FPAD}
  , conv3_2{N_F3, N_F3, FSIZE, FSIZE, FSTRID, FPAD}
  , conv3_3{N_F3, N_F3, FSIZE, FSIZE, FSTRID, FPAD}
  , conv4_1{N_F4, N_F3, FSIZE, FSIZE, FSTRID, FPAD}
  , conv4_2{N_F4, N_F4, FSIZE, FSIZE, FSTRID, FPAD}
  , conv4_3{N_F4, N_F4, FSIZE, FSIZE, FSTRID, FPAD}
  , conv5_1{N_F5, N_F4, FSIZE, FSIZE, FSTRID, FPAD}
  , conv5_2{N_F5, N_F5, FSIZE, FSIZE, FSTRID, FPAD}
  , conv5_3{N_F5, N_F5, FSIZE, FSIZE, FSTRID, FPAD}
  , pool1{PSIZE, PSIZE, PSTRID}
  , pool2{PSIZE, PSIZE, PSTRID}
  , pool3{PSIZE, PSIZE, PSTRID}
  , pool4{PSIZE, PSIZE, PSTRID}
  , pool5{PSIZE, PSIZE, PSTRID}
#endif
  , full6{N_H1, N_F5*pm5hei*pm5wid}
  , full7{N_H2, N_H1}
  , full8{LABEL, N_H2}
{
  input = zeros<T>(   3, INSIZE, INSIZE);

  fmap1_1 = zeros<T>(N_F1, fm1hei, fm1wid);
  amap1_1 = zeros<T>(N_F1, fm1hei, fm1wid);
  fmap1_2 = zeros<T>(N_F1, fm1hei, fm1wid);
  pmap1   = zeros<T>(N_F1, pm1hei, pm1wid);
  amap1_2 = zeros<T>(N_F1, pm1hei, pm1wid);

  fmap2_1 = zeros<T>(N_F2, pm1hei, pm1wid);
  amap2_1 = zeros<T>(N_F2, pm1hei, pm1wid);
  fmap2_2 = zeros<T>(N_F2, pm1hei, pm1wid);
  pmap2   = zeros<T>(N_F2, pm2hei, pm2wid);
  amap2_2 = zeros<T>(N_F2, pm2hei, pm2wid);

  fmap3_1 = zeros<T>(N_F3, pm2hei, pm2wid);
  amap3_1 = zeros<T>(N_F3, pm2hei, pm2wid);
  fmap3_2 = zeros<T>(N_F3, pm2hei, pm2wid);
  amap3_2 = zeros<T>(N_F3, pm2hei, pm2wid);
  fmap3_3 = zeros<T>(N_F3, pm2hei, pm2wid);
  pmap3   = zeros<T>(N_F3, pm3hei, pm3wid);
  amap3_3 = zeros<T>(N_F3, pm3hei, pm3wid);

  fmap4_1 = zeros<T>(N_F4, pm3hei, pm3wid);
  amap4_1 = zeros<T>(N_F4, pm3hei, pm3wid);
  fmap4_2 = zeros<T>(N_F4, pm3hei, pm3wid);
  amap4_2 = zeros<T>(N_F4, pm3hei, pm3wid);
  fmap4_3 = zeros<T>(N_F4, pm3hei, pm3wid);
  pmap4   = zeros<T>(N_F4, pm4hei, pm4wid);
  amap4_3 = zeros<T>(N_F4, pm4hei, pm4wid);

  fmap5_1 = zeros<T>(N_F5, pm4hei, pm4wid);
  amap5_1 = zeros<T>(N_F5, pm4hei, pm4wid);
  fmap5_2 = zeros<T>(N_F5, pm4hei, pm4wid);
  amap5_2 = zeros<T>(N_F5, pm4hei, pm4wid);
  fmap5_3 = zeros<T>(N_F5, pm4hei, pm4wid);
  pmap5   = zeros<T>(N_F5, pm5hei, pm5wid);
  amap5_3 = zeros<T>(N_F5, pm5hei, pm5wid);

  amap5_flat = zeros<T>(N_F5*pm5hei*pm5wid);
  hunit1 = zeros<T>(N_H1);
  aunit1 = zeros<T>(N_H1);
  hunit2 = zeros<T>(N_H2);
  aunit2 = zeros<T>(N_H2);
  output = zeros<T>(LABEL);
}

template <typename T>
VGGNet<T>::~VGGNet()
{
}

template <typename T>
void VGGNet<T>::Load(string path)
{
#if defined _EAGER
  conv1_1.load(path+"/wb_1_1");
  conv1_2.load(path+"/wb_1_2");
  std::cout << "1" << std::endl;
  conv2_1.load(path+"/wb_2_1");
  conv2_2.load(path+"/wb_2_2");
  std::cout << "2" << std::endl;
  conv3_1.load(path+"/wb_3_1");
  conv3_2.load(path+"/wb_3_2");
  conv3_3.load(path+"/wb_3_3");
  std::cout << "3" << std::endl;
  conv4_1.load(path+"/wb_4_1");
  conv4_2.load(path+"/wb_4_2");
  conv4_3.load(path+"/wb_4_3");
  std::cout << "4" << std::endl;
  conv5_1.load(path+"/wb_5_1");
  conv5_2.load(path+"/wb_5_2");
  conv5_3.load(path+"/wb_5_3");
  std::cout << "5" << std::endl;
#elif defined _LAZY
  conv1_1.load(path+"/wb_1_1");
  conv1_2.load(path+"/wb_1_2");
  //convpool1.load(filename);
  conv2_1.load(path+"/wb_2_1");
  conv2_2.load(path+"/wb_2_2");
  //convpool2.load(filename);
  conv3_1.load(path+"/wb_3_1");
  conv3_2.load(path+"/wb_3_2");
  conv3_3.load(path+"/wb_3_3");
  //convpool3.load(filename);
  conv4_1.load(path+"/wb_4_1");
  conv4_2.load(path+"/wb_4_2");
  conv4_3.load(path+"/wb_4_3");
  //convpool4.load(filename);
  conv5_1.load(path+"/wb_5_1");
  conv5_2.load(path+"/wb_5_2");
  conv5_3.load(path+"/wb_5_3");
  //convpool5.load(filename);
#endif
  full6.load(path+"/wb_6");
  std::cout << "6" << std::endl;
  full7.load(path+"/wb_7");
  std::cout << "7" << std::endl;
  full8.load(path+"/wb_8");
  std::cout << "8" << std::endl;
}

template <typename T>
vector<int> VGGNet<T>::calc(string data)
{
  // Top-5 label
  vector<int> number(5, -1);

  load_image(data, input);

#if defined _EAGER
  conv1_1.forward(  input, fmap1_1);
  relu1_1.forward(fmap1_1, amap1_1);
  conv1_2.forward(amap1_1, fmap1_2);
    pool1.forward(fmap1_2, pmap1);
  relu1_2.forward(pmap1, amap1_2);

  conv2_1.forward(amap1_2, fmap2_1);
  relu2_1.forward(fmap2_1, amap2_1);
  conv2_2.forward(amap2_1, fmap2_2);
    pool2.forward(fmap2_2, pmap2);
  relu2_2.forward(pmap2, amap2_2);

  conv3_1.forward(amap2_2, fmap3_1);
  relu3_1.forward(fmap3_1, amap3_1);
  conv3_2.forward(amap3_1, fmap3_2);
  relu3_2.forward(fmap3_2, amap3_2);
  conv3_3.forward(amap3_2, fmap3_3);
    pool3.forward(fmap3_3, pmap3);
  relu3_3.forward(pmap3, amap3_3);

  conv4_1.forward(amap3_3, fmap4_1);
  relu4_1.forward(fmap4_1, amap4_1);
  conv4_2.forward(amap4_1, fmap4_2);
  relu4_2.forward(fmap4_2, amap4_2);
  conv4_3.forward(amap4_2, fmap4_3);
    pool4.forward(fmap4_3, pmap4);
  relu4_3.forward(pmap4, amap4_3);

  conv5_1.forward(amap4_3, fmap5_1);
  relu5_1.forward(fmap5_1, amap5_1);
  conv5_2.forward(amap5_1, fmap5_2);
  relu5_2.forward(fmap5_2, amap5_2);
  conv5_3.forward(amap5_2, fmap5_3);
    pool5.forward(fmap5_3, pmap5);
  relu5_3.forward(pmap5, amap5_3);

#elif defined _LAZY
  const int which = 0;
  const int amount = 0;
  conv1_1.forward(  input, fmap1_1);
  relu1_1.forward(fmap1_1, amap1_1);
  conv1_2.forward(amap1_1, fmap1_2);
    pool1.forward(fmap1_2, pmap1);
  relu1_2.forward(pmap1, amap1_2);
  conv2_1.forward(amap1_2, fmap2_1);
  relu2_1.forward(fmap2_1, amap2_1);
  conv2_2.forward(amap2_1, fmap2_2);
    pool2.forward(fmap2_2, pmap2);
  relu2_2.forward(pmap2, amap2_2);
  conv3_1.forward(amap2_2, fmap3_1);
  relu3_1.forward(fmap3_1, amap3_1);
  conv3_2.forward(amap3_1, fmap3_2);
  relu3_2.forward(fmap3_2, amap3_2);
  conv3_3.forward(amap3_2, fmap3_3);
    pool3.forward(fmap3_3, pmap3);
  relu3_3.forward(pmap3, amap3_3);
  conv4_1.forward(amap3_3, fmap4_1);
  relu4_1.forward(fmap4_1, amap4_1);
  conv4_2.forward(amap4_1, fmap4_2);
  relu4_2.forward(fmap4_2, amap4_2);
  conv4_3.forward(amap4_2, fmap4_3);
    pool4.forward(fmap4_3, pmap4);
  relu4_3.forward(pmap4, amap4_3);
  conv5_1.forward(amap4_3, fmap5_1);
  relu5_1.forward(fmap5_1, amap5_1);
  conv5_2.forward(amap5_1, fmap5_2);
  relu5_2.forward(fmap5_2, amap5_2);
  conv5_3.forward(amap5_2, fmap5_3);
    pool5.forward(fmap5_3, pmap5);
  relu5_3.forward(pmap5, amap5_3);
#endif

  flatten(amap5_3, amap5_flat, N_F5, pm5hei, pm5wid);

  full6.forward(amap5_flat, hunit1);
  relu6.forward(hunit1, aunit1);
  full7.forward(aunit1, hunit2);
  relu7.forward(hunit2, aunit2);

  full8.forward(aunit2, output);

  //output4.forward(output);

  vector<int> index(LABEL);
  iota(index.begin(), index.end(), 0);
  sort(index.begin(), index.end(), [&](int a, int b){
    return output[a] > output[b];
  });
  for (int i=0; i<5; i++) number[i] = index[i];

  return number;
}

#endif
