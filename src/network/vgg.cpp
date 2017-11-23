#ifdef _VGG_HPP_

template <typename T>
VGG<T>::VGG()
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
VGG<T>::~VGG()
{
}

template <typename T>
void VGG<T>::Load(std::string path)
{
  conv1_1.load(path+"/wb_1_1");
  conv1_2.load(path+"/wb_1_2");
  conv2_1.load(path+"/wb_2_1");
  conv2_2.load(path+"/wb_2_2");
  conv3_1.load(path+"/wb_3_1");
  conv3_2.load(path+"/wb_3_2");
  conv3_3.load(path+"/wb_3_3");
  conv4_1.load(path+"/wb_4_1");
  conv4_2.load(path+"/wb_4_2");
  conv4_3.load(path+"/wb_4_3");
  conv5_1.load(path+"/wb_5_1");
  conv5_2.load(path+"/wb_5_2");
  conv5_3.load(path+"/wb_5_3");
  full6.load(path+"/wb_6");
  full7.load(path+"/wb_7");
  full8.load(path+"/wb_8");
}

template <typename T>
std::vector<int> VGG<T>::calc(std::string data)
{
  load_txt(input, data);

  conv1_1.forward(fmap1_1,   input);
  relu1_1.forward(amap1_1, fmap1_1);
  conv1_2.forward(fmap1_2, amap1_1);
    pool1.forward(pmap1, fmap1_2);
  relu1_2.forward(amap1_2, pmap1);

  conv2_1.forward(fmap2_1, amap1_2);
  relu2_1.forward(amap2_1, fmap2_1);
  conv2_2.forward(fmap2_2, amap2_1);
    pool2.forward(pmap2, fmap2_2);
  relu2_2.forward(amap2_2, pmap2);

  conv3_1.forward(fmap3_1, amap2_2);
  relu3_1.forward(amap3_1, fmap3_1);
  conv3_2.forward(fmap3_2, amap3_1);
  relu3_2.forward(amap3_2, fmap3_2);
  conv3_3.forward(fmap3_3, amap3_2);
    pool3.forward(pmap3, fmap3_3);
  relu3_3.forward(amap3_3, pmap3);

  conv4_1.forward(fmap4_1, amap3_3);
  relu4_1.forward(amap4_1, fmap4_1);
  conv4_2.forward(fmap4_2, amap4_1);
  relu4_2.forward(amap4_2, fmap4_2);
  conv4_3.forward(fmap4_3, amap4_2);
    pool4.forward(pmap4, fmap4_3);
  relu4_3.forward(amap4_3, pmap4);

  conv5_1.forward(fmap5_1, amap4_3);
  relu5_1.forward(amap5_1, fmap5_1);
  conv5_2.forward(fmap5_2, amap5_1);
  relu5_2.forward(amap5_2, fmap5_2);
  conv5_3.forward(fmap5_3, amap5_2);
    pool5.forward(pmap5, fmap5_3);
  relu5_3.forward(amap5_3, pmap5);

  flatten(amap5_flat, amap5_3);

  full6.forward(hunit1, amap5_flat);
  relu6.forward(aunit1, hunit1);
  full7.forward(hunit2, aunit1);
  relu7.forward(aunit2, hunit2);

  full8.forward(output, aunit2);
  //output4.forward(output);

  return classify_top(output, 5);
}

#endif
